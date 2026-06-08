Recommended structure

Build this as an experiment factory, not as a single cloud VM project. The right shape is:

Local machine = control plane.
You edit code, run CPU smoke tests, review agent proposals, inspect results, and maintain the research ledger locally.

Cloud GPUs = disposable workers.
A GPU machine should pull an exact Git commit, run one declared experiment or sweep shard, upload logs/artifacts/results, and shut itself down.

Repo = source of truth.
All hypotheses, configs, launcher specs, dataset-generation code, metrics, dashboards, and postmortems live in the repo. No important state should exist only on a VM.

This matches the research problem: catapult/grokking experiments require many carefully logged schedule/data/seed variations, while Guardian Angel work needs append-only logs, dynamic evaluation, active elicitation, and controlled personalization loops. The source essays explicitly frame catapulting around cyclical/high-LR schedules, hard held-out benchmarks, arithmetic/image stress tests, and dynamic evaluation/dynamic grokking; the Guardian Angel post frames personalization around append-only logs, active learning, PRINCIPAL.md, Q&A logs, and repeated reprocessing/finetuning.

Answering your specific workflow questions

Should an agent create the repo?
Yes, but have it create a skeleton plus runnable smoke tests first, not a giant speculative codebase. The first milestone should be: “I can run a tiny arithmetic/grokking experiment locally on CPU, log metrics, produce a report, and dry-run a cloud launch command.”

Should you clone it onto a cloud VM and run GPU jobs there?
For development, no. For execution, yes in a controlled way. Avoid SSHing into a GPU VM and editing files there. Instead, use cloud workers that run:

git clone <repo>
git checkout <exact_sha>
uv sync
python -m guardian_research.train +exp=arithmetic_catapult seed=3

Use GPU VMs as cattle, not pets.

Can the repo kick off cloud training runs automatically while tracking stays local?
Yes, but with one caveat: cloud GPUs should not depend on your laptop being awake or reachable. The robust pattern is:

Local CLI launches a job.
Cloud job writes metrics locally during the run.
Cloud job periodically uploads checkpoints, metrics, logs, and results.json.
Local machine syncs/ingests results into your local dashboard.

If you want live tracking, use a small always-on tracking endpoint, not your laptop. MLflow supports a tracking UI for runs, params, metrics, artifacts, and run comparison, and it can use local, database-backed, or remote tracking stores plus artifact stores such as S3/GCS/Azure/NFS.

Can agents read experiment results and launch new ones autonomously?
Yes, but not with unrestricted cloud permissions. Use a proposal/approval loop:

results → analyzer agent → next_sweep_proposal.yaml → validator → human approval → launcher

Allow full autonomy only for CPU tests and tiny budget-capped GPU probes. Anything that can spend serious money, touch private data, delete artifacts, or change infra should require approval.

Minimal stack I would use

Use SkyPilot as the first cloud launcher. It gives you a unified interface for clusters, jobs, services, GPU workloads, finetuning, sweeps, batch jobs, and managed jobs; managed jobs run in temporary clusters with auto-recovery, and SkyPilot also supports autostop and managed spot usage. RunPod also documents SkyPilot integration with a YAML workdir, setup, and run structure, which is exactly the workflow you want for repo-driven experiments.

Use Hydra for configuration and sweeps. Hydra supports multirun sweeps from config or command line, which is ideal for exploring LR schedules, weight decay cycles, seed variance, model sizes, and hard-set filtering.

Use MLflow + DVC by default. MLflow should track metrics, params, checkpoints, plots, and artifacts. DVC should version generated datasets, hard splits, derived corpora, and report artifacts. DVC is designed to version data/models with Git metadata while storing large assets elsewhere, and it supports experiment tracking, metrics, plots, and reproducible pipelines.

Use W&B only if you are comfortable with the data going to a managed service, or if you self-host/use an enterprise setup. W&B Sweeps can automate hyperparameter search across machines, W&B Artifacts can version datasets and model outputs, and W&B Launch can create jobs from Git commits, but Guardian Angel work will quickly involve sensitive user/personality data, so I would default to local/self-hosted tracking until the security model is mature. The GA post itself emphasizes that serious GA systems involve sensitive personal data and should be designed around strong security rather than casual cloud SaaS assumptions.

Repo layout
guardian-research/
  README.md
  pyproject.toml
  uv.lock
  justfile                      # reproducible command aliases
  .env.example
  .gitignore

  planning/
    guardian/
      llm-catapult.md
      guardian-angel.md
      research-program.md
    hypotheses/
      H001-catapult-arithmetic.md
      H002-dynamic-eval-persona.md
    decisions/
      ADR-0001-experiment-factory.md

  conf/
    config.yaml
    exp/
      arithmetic_catapult.yaml
      cifar_robustness.yaml
      dynamic_grokking.yaml
      persona_lora.yaml
      persona_dynamic_eval.yaml
    model/
      tiny_transformer.yaml
      nano_gpt.yaml
      resnet_cifar.yaml
      mlp_mixer_cifar.yaml
    schedule/
      baseline_cosine.yaml
      onecycle_high_lr.yaml
      cyclic_lr.yaml
      cyclic_weight_decay.yaml
    launcher/
      local.yaml
      skypilot_runpod.yaml
      skypilot_lambda.yaml
    sweep/
      arith_lr_wd_seed_v0.yaml
      dynamic_eval_steps_v0.yaml

  src/guardian_research/
    cli.py                      # `ga ...`
    common/
      config.py
      logging.py
      seeding.py
      artifacts.py
      budget.py
    launchers/
      local.py
      skypilot.py
      modal.py                  # optional later
    tracking/
      mlflow_client.py
      ingest.py
      reports.py
    data/
      arithmetic.py
      hardset_mining.py
      persona_corpus.py
      dvc_utils.py
    experiments/
      arithmetic_catapult/
        train.py
        eval.py
        analyze.py
      cifar_robustness/
        train.py
        eval_robustness.py
      dynamic_grokking/
        run.py
        dynamic_eval.py
      persona/
        prepare_corpus.py
        train_lora.py
        eval_persona.py
        active_questions.py
    agents/
      summarize_results.py
      propose_sweeps.py
      validate_proposal.py

  infra/
    docker/
      Dockerfile.gpu
      Dockerfile.cpu
    sky/
      train.yaml
      sweep.yaml
      eval.yaml
    mlflow/
      docker-compose.yml

  data/
    README.md
    raw/                        # DVC-tracked or gitignored
    processed/                  # DVC-tracked
    splits/                     # DVC-tracked

  runs/                         # gitignored local cache
  artifacts/                    # gitignored local cache

  reports/
    index.md
    latest.md
    figures/
    runs/
    proposals/

  tests/
    test_arithmetic_data.py
    test_config_loads.py
    test_smoke_train_cpu.py
    test_budget_guard.py
    test_result_schema.py
Command surface

The repo should expose a small CLI so agents do not invent new workflows every time.

# local setup
just setup
just test
just smoke

# generate data
ga data arithmetic --digits-train 2 --digits-test 3 --hard-filter carry_chain --out data/processed/arith_v0

# run locally
ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0
ga train +exp=arithmetic_catapult model=tiny_transformer schedule=onecycle_high_lr seed=0

# dry-run cloud launch
ga launch --dry-run --provider skypilot +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0

# launch a budget-capped cloud sweep
ga launch --provider skypilot --max-cost-usd 25 +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0

# monitor and collect
ga status
ga logs <run_id>
ga cancel <run_id>
ga collect <run_id>

# analyze
ga analyze --experiment arithmetic_catapult --since 7d --write reports/runs/arith_v0.md

# agent loop
ga propose --experiment arithmetic_catapult --budget-usd 50 --write reports/proposals/next_arith_sweep.yaml
ga validate-proposal reports/proposals/next_arith_sweep.yaml
ga approve reports/proposals/next_arith_sweep.yaml
Autonomy model

Use three autonomy tiers.

Tier 0: unrestricted.
Agents may edit docs, add tests, run CPU smoke tests, summarize results, and propose next experiments.

Tier 1: bounded automation.
Agents may launch small GPU jobs only if all of these are true:

max_job_cost_usd: 5
max_daily_cost_usd: 25
allowed_data_classes: ["public", "synthetic"]
allowed_providers: ["runpod", "lambda", "modal"]
allow_private_persona_data: false
require_clean_git_tree: true
require_exact_commit_sha: true
require_dry_run_first: true

Tier 2: approval required.
Any job using private data, more than $5, checkpoint uploads, new cloud credentials, deletion, model publishing, or more than one GPU requires explicit approval.

This gives you the benefit of autonomous iteration without letting an LLM accidentally burn the $2,000 budget or leak the corpus that is supposed to become the basis of a personalized model.

First build milestones

Milestone 1: experiment factory skeleton.
The repo can run just smoke, create a synthetic arithmetic dataset, train a tiny Transformer on CPU for 100 steps, log metrics, and render a local report.

Milestone 2: cloud execution.
ga launch --dry-run prints the exact SkyPilot YAML and estimated resources. ga launch --provider skypilot --max-cost-usd 5 runs a tiny GPU job and returns logs/artifacts.

Milestone 3: catapult arithmetic.
Implement baseline cosine, one-cycle/high-LR, cyclic LR, cyclic WD, and high-WD recipes. Compare easy vs hard arithmetic splits, train/test memorization gap, loss spikes, checkpoint trajectory, and “curves cross” behavior.

Milestone 4: dynamic grokking harness.
Implement repeated dynamic evaluation on a hard prompt/problem: update model/LoRA/adapters for N inner steps, sample outputs, optionally perform “sleep” regularization, and compare against brute-force sampling at equal cost. This maps directly onto the dynamic-grokking proposal in the catapult essay.

Milestone 5: Guardian Angel toy.
Use public or synthetic persona corpora first. Create PRINCIPAL.md, per-document summaries, Q&A logs, and an active-question generator. Evaluate base vs RAG vs LoRA vs dynamic-eval personalization on held-out “what would the principal prefer/write/choose?” tasks. This should mirror the GA post’s proposed PRINCIPAL.md, Q&A logs, annotations, active learning, and retraining loop.