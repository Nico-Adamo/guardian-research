"""Dynamic-grokking harness: "pondering" via dynamic evaluation.

The idea (a *toy*, not a claim): a base model is trained quickly on EASY
arithmetic and will get many HARD (out-of-distribution / carry-heavy) problems
wrong. At inference time we can spend extra compute in one of two ways and ask
which one is a better use of the same budget:

  * **dynamic** — for each hard problem, take a handful of inner gradient steps
    that adapt the model to *that single problem's* prompt (next-token
    "dynamic evaluation" / test-time training), sampling candidate answers
    between inner steps and stopping early on a correct one. Optionally a
    "sleep" step shrinks weights (extra decay) between problems so adaptation
    to one problem does not corrupt the base.
  * **static** — spend the SAME number of sampled tokens on plain temperature
    sampling from the frozen base model (brute-force best-of-N).

We record per-inner-step loss and candidate correctness, output DIVERSITY
(unique candidate count), and a final static-vs-dynamic comparison at EQUAL
compute. It is entirely expected at this toy scale that dynamic does NOT beat
static; the harness reports whatever happens, honestly.

The weight-update *target* is modular (``dynamic.update_target``):
  * ``"full"``       — adapt all parameters (works now).
  * ``"last_layer"`` — adapt only the final LM head + last block (works now).
  * ``"lora"``       — honest stub; raises NotImplementedError with guidance.

Everything runs on CPU in well under a minute at the configured toy sizes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ...common.artifacts import RunWriter, new_run_id
from ...common.logging import get_logger
from ...common.seeding import seed_everything
from ...data.arithmetic import ArithmeticConfig, CharTokenizer, build_splits
from ...models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from ...tracking.mlflow_client import start_run

log = get_logger(__name__)

UPDATE_TARGETS = ("full", "last_layer", "lora")


def _resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # MPS skipped by default for determinism


def _masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    b, t, v = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(b * t, v), targets.reshape(b * t), reduction="none"
    ).reshape(b, t)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def _train_base(
    model: TinyTransformer,
    loader: DataLoader,
    device: str,
    max_steps: int,
    lr: float,
    weight_decay: float,
) -> float:
    """Quickly train the base model on EASY data. Returns the last train loss."""
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    model.train()
    last = 0.0
    step = 0
    data_iter = iter(loader)
    while step < max_steps:
        try:
            x, y, m = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y, m = next(data_iter)
        x, y, m = x.to(device), y.to(device), m.to(device)
        optim.zero_grad(set_to_none=True)
        loss = _masked_ce(model(x), y, m)
        loss.backward()
        optim.step()
        last = float(loss.item())
        step += 1
    return last


@torch.no_grad()
def _sample_answer(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str,
    max_new: int,
    temperature: float,
    generator: torch.Generator,
    device: str,
) -> tuple[str, int]:
    """Temperature-sample one answer. Returns (decoded_answer, tokens_generated).

    ``tokens_generated`` is the number of forward passes used (the unit of
    compute we equalize between static and dynamic).
    """
    model.eval()
    ids = [tok.bos_id] + tok.encode(prompt)
    start = len(ids)
    max_len = model.cfg.max_len
    for _ in range(max_new):
        window = ids[-max_len:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        if temperature <= 0:
            nxt = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        ids.append(nxt)
        if nxt == tok.eos_id:
            break
    generated = ids[start:]
    return tok.decode(generated), len(generated)


def _single_problem_batch(
    tok: CharTokenizer, prompt: str, answer: str, max_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a 1-example (input, target, answer-only-mask) batch for one problem.

    This is the dynamic-eval supervision: we adapt the model toward producing
    *this problem's* answer. (At true test time the answer is unknown; here the
    toy uses the known answer as the adaptation target to study the mechanism.)
    """
    seq = [tok.bos_id] + tok.encode(prompt + answer) + [tok.eos_id]
    seq = seq[:max_len]
    prompt_len = 1 + len(prompt)  # bos + prompt chars (the '=' is the last prompt char)
    input_ids = seq[:-1]
    target_ids = seq[1:]
    mask = [1.0 if (i + 1) >= prompt_len else 0.0 for i in range(len(target_ids))]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    y = torch.tensor([target_ids], dtype=torch.long, device=device)
    m = torch.tensor([mask], dtype=torch.float, device=device)
    return x, y, m


def _select_inner_params(model: TinyTransformer, update_target: str) -> list[torch.nn.Parameter]:
    """Pick the parameters the inner (dynamic-eval) optimizer is allowed to move."""
    if update_target == "full":
        return [p for p in model.parameters() if p.requires_grad]
    if update_target == "last_layer":
        # Adapt only the final block + final layernorm + LM head. With tied
        # weights the head shares the token embedding, so this also nudges
        # embeddings; that is fine for a toy and kept intentionally simple.
        params: list[torch.nn.Parameter] = []
        if len(model.blocks) > 0:
            params += list(model.blocks[-1].parameters())
        params += list(model.ln_f.parameters())
        params += list(model.lm_head.parameters())
        # De-duplicate (tied weights appear twice) preserving order.
        seen: set[int] = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique
    if update_target == "lora":
        raise NotImplementedError(
            "update_target='lora' is not implemented yet. The harness is structured "
            "so a LoRA/adapter module can be injected and its parameters returned here; "
            "use 'full' or 'last_layer' for now."
        )
    raise ValueError(f"unknown update_target '{update_target}'. Known: {sorted(UPDATE_TARGETS)}")


def _dynamic_solve(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str,
    answer: str,
    *,
    update_target: str,
    inner_steps: int,
    inner_lr: float,
    inner_wd: float,
    samples_per_step: int,
    temperature: float,
    max_new: int,
    sleep_decay: float,
    generator: torch.Generator,
    device: str,
    log_prefix: str,
    writer: RunWriter,
    log_step0: int,
) -> dict[str, Any]:
    """Adapt a *copy* of the model to one problem; sample between inner steps.

    Returns a dict with success flag, tokens spent, per-inner-step loss series,
    and the set of unique candidate strings seen (for diversity).
    """
    # Work on a copy so each problem starts from the same frozen base and one
    # problem's adaptation never leaks into the next.
    local = copy.deepcopy(model).to(device)
    inner_params = _select_inner_params(local, update_target)
    optim = torch.optim.AdamW(inner_params, lr=inner_lr, weight_decay=inner_wd, betas=(0.9, 0.98))

    max_len = local.cfg.max_len
    x, y, m = _single_problem_batch(tok, prompt, answer, max_len, device)

    candidates: set[str] = set()
    loss_series: list[float] = []
    tokens_spent = 0
    success = False
    solved_at_step = -1

    # inner_steps+1 sampling rounds: one before any adaptation, then one after
    # each inner gradient step.
    for inner in range(inner_steps + 1):
        # Sample candidates at the current weights.
        for _ in range(samples_per_step):
            cand, n_tok = _sample_answer(
                local, tok, prompt, max_new, temperature, generator, device
            )
            candidates.add(cand)
            tokens_spent += n_tok
            if cand == answer and not success:
                success = True
                solved_at_step = inner
        if success:
            break  # stop on success (compute budget saved)

        # One inner adaptation step (dynamic evaluation / test-time training).
        local.train()
        optim.zero_grad(set_to_none=True)
        loss = _masked_ce(local(x), y, m)
        loss.backward()
        optim.step()
        loss_series.append(float(loss.item()))
        writer.log_metrics(log_step0 + inner, **{f"{log_prefix}_inner_loss": float(loss.item())})

        # Optional "sleep": shrink the adapted weights a touch toward zero. A
        # crude analogue of consolidation / extra weight decay between problems.
        if sleep_decay > 0:
            with torch.no_grad():
                for p in inner_params:
                    p.mul_(1.0 - sleep_decay)

    return {
        "success": success,
        "tokens_spent": tokens_spent,
        "loss_series": loss_series,
        "n_unique": len(candidates),
        "solved_at_step": solved_at_step,
    }


def _static_solve(
    model: TinyTransformer,
    tok: CharTokenizer,
    prompt: str,
    answer: str,
    *,
    token_budget: int,
    temperature: float,
    max_new: int,
    generator: torch.Generator,
    device: str,
) -> dict[str, Any]:
    """Brute-force best-of-N from the FROZEN base, capped at ``token_budget``.

    Equal-compute baseline: keep sampling until either a correct answer appears
    or the token budget (the dynamic run's forward-pass count) is exhausted.
    """
    candidates: set[str] = set()
    tokens_spent = 0
    success = False
    n_samples = 0
    while tokens_spent < token_budget:
        cand, n_tok = _sample_answer(model, tok, prompt, max_new, temperature, generator, device)
        candidates.add(cand)
        tokens_spent += n_tok
        n_samples += 1
        if cand == answer:
            success = True
            break
    return {
        "success": success,
        "tokens_spent": tokens_spent,
        "n_unique": len(candidates),
        "n_samples": n_samples,
    }


@torch.no_grad()
def _greedy_correct(model: TinyTransformer, tok: CharTokenizer, prompt: str, answer: str, max_new: int) -> bool:
    prompt_ids = torch.tensor([tok.bos_id] + tok.encode(prompt), dtype=torch.long)
    gen = model.generate_greedy(prompt_ids, max_new_tokens=max_new, eos_id=tok.eos_id)
    return tok.decode(gen) == answer


def run(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = _resolve_device(str(cfg.get("device", "auto")))

    data_cfg_dict = cfg.get("data", {})
    model_cfg_dict = cfg.get("model", {})
    tr = cfg.get("train", {})
    dyn = cfg.get("dynamic", {})

    update_target = str(dyn.get("update_target", "full"))
    if update_target not in UPDATE_TARGETS:
        raise ValueError(f"unknown update_target '{update_target}'. Known: {sorted(UPDATE_TARGETS)}")

    inner_steps = int(dyn.get("inner_steps", 8))
    inner_lr = float(dyn.get("inner_lr", 1e-2))
    inner_wd = float(dyn.get("inner_wd", 0.0))
    samples_per_step = int(dyn.get("samples_per_step", 2))
    temperature = float(dyn.get("temperature", 0.8))
    sleep_decay = float(dyn.get("sleep_decay", 0.0))
    n_problems = int(dyn.get("n_problems", 16))

    # -- data ------------------------------------------------------------- #
    acfg = ArithmeticConfig(
        **{k: v for k, v in data_cfg_dict.items() if k in ArithmeticConfig.__dataclass_fields__},
        seed=seed,
    )
    splits = build_splits(acfg)
    tok: CharTokenizer = splits["tokenizer"]
    max_len = splits["max_len"]
    easy_ds, hard_ds = splits["easy_eval"], splits["hard_eval"]
    train_ds = splits["train"]
    max_new = max_len  # generous decode budget; <eos> usually stops earlier

    batch_size = int(tr.get("batch_size", 128))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # -- base model ------------------------------------------------------- #
    mcfg = TinyTransformerConfig(
        vocab_size=tok.vocab_size,
        max_len=max_len,
        d_model=int(model_cfg_dict.get("d_model", 64)),
        n_layers=int(model_cfg_dict.get("n_layers", 3)),
        n_heads=int(model_cfg_dict.get("n_heads", 4)),
        d_ff=int(model_cfg_dict.get("d_ff", 256)),
        dropout=float(model_cfg_dict.get("dropout", 0.0)),
        tie_weights=bool(model_cfg_dict.get("tie_weights", True)),
    )
    model = TinyTransformer(mcfg).to(device)

    base_steps = int(tr.get("max_steps", 300))
    base_lr = float(tr.get("lr", 3e-3))
    base_wd = float(tr.get("weight_decay", 0.05))

    experiment = str(cfg.get("experiment", "dynamic_grokking"))
    run_id = new_run_id("dyngrok")
    writer = RunWriter(experiment, run_id, seed)
    writer.set_config(cfg)
    writer.set_params(
        update_target=update_target,
        inner_steps=inner_steps,
        inner_lr=inner_lr,
        samples_per_step=samples_per_step,
        temperature=temperature,
        sleep_decay=sleep_decay,
        n_problems=n_problems,
        base_steps=base_steps,
        num_params=float(model.num_params()),
        device=device,
    )

    log.info(
        "dynamic_grokking: target=%s inner_steps=%d samples/step=%d n_problems=%d base_steps=%d device=%s",
        update_target, inner_steps, samples_per_step, n_problems, base_steps, device,
    )

    with start_run(experiment, run_id, tags={"update_target": update_target, "seed": str(seed)}) as mlf:
        mlf.log_params({**writer.result.params})
        try:
            # 1) Quick base train on EASY data.
            base_loss = _train_base(model, train_loader, device, base_steps, base_lr, base_wd)
            writer.log_metrics(0, base_train_loss=base_loss)

            # 2) Pick HARD problems the FROZEN base fails (greedy). These are the
            #    candidates where extra test-time compute might help.
            hard_problems: list[tuple[str, str]] = []
            idx = 0
            while idx < len(hard_ds) and len(hard_problems) < n_problems:
                prompt, answer = hard_ds.text(idx)
                if not _greedy_correct(model, tok, prompt, answer, max_new):
                    hard_problems.append((prompt, answer))
                idx += 1
            # Fall back to easy-but-failed if the base happened to solve all hard
            # problems greedily (keeps the toy producing a comparison regardless).
            ej = 0
            while len(hard_problems) < n_problems and ej < len(easy_ds):
                prompt, answer = easy_ds.text(ej)
                if not _greedy_correct(model, tok, prompt, answer, max_new):
                    hard_problems.append((prompt, answer))
                ej += 1

            writer.set_params(n_failed_problems=len(hard_problems))

            # 3) For each problem: dynamic solve, then static solve at EQUAL
            #    compute (the dynamic run's token count is the static budget).
            dyn_successes = 0
            static_successes = 0
            dyn_tokens_total = 0
            static_tokens_total = 0
            dyn_unique_total = 0
            static_unique_total = 0
            # Two independent RNG streams so static vs dynamic sampling are
            # comparable but not identical.
            gen_dyn = torch.Generator(device=device).manual_seed(seed + 101)
            gen_static = torch.Generator(device=device).manual_seed(seed + 202)

            for i, (prompt, answer) in enumerate(hard_problems):
                d = _dynamic_solve(
                    model, tok, prompt, answer,
                    update_target=update_target,
                    inner_steps=inner_steps,
                    inner_lr=inner_lr,
                    inner_wd=inner_wd,
                    samples_per_step=samples_per_step,
                    temperature=temperature,
                    max_new=max_new,
                    sleep_decay=sleep_decay,
                    generator=gen_dyn,
                    device=device,
                    log_prefix=f"prob{i}",
                    writer=writer,
                    log_step0=i * (inner_steps + 1),
                )
                s = _static_solve(
                    model, tok, prompt, answer,
                    token_budget=max(d["tokens_spent"], max_new),
                    temperature=temperature,
                    max_new=max_new,
                    generator=gen_static,
                    device=device,
                )
                dyn_successes += int(d["success"])
                static_successes += int(s["success"])
                dyn_tokens_total += d["tokens_spent"]
                static_tokens_total += s["tokens_spent"]
                dyn_unique_total += d["n_unique"]
                static_unique_total += s["n_unique"]
                writer.log_metrics(
                    i,
                    dyn_success=float(d["success"]),
                    static_success=float(s["success"]),
                    dyn_tokens=float(d["tokens_spent"]),
                    static_tokens=float(s["tokens_spent"]),
                    dyn_unique=float(d["n_unique"]),
                    static_unique=float(s["n_unique"]),
                )
                mlf.log_metrics(
                    {"dyn_success": float(d["success"]), "static_success": float(s["success"])}, step=i
                )

            n = max(1, len(hard_problems))
            dynamic_success = dyn_successes / n
            static_success = static_successes / n
            final = dict(
                n_problems=float(len(hard_problems)),
                dynamic_success=dynamic_success,
                static_success=static_success,
                dynamic_minus_static=dynamic_success - static_success,
                dynamic_solved=float(dyn_successes),
                static_solved=float(static_successes),
                dynamic_tokens_total=float(dyn_tokens_total),
                static_tokens_total=float(static_tokens_total),
                dynamic_unique_mean=dyn_unique_total / n,
                static_unique_mean=static_unique_total / n,
                base_train_loss=base_loss,
                num_params=float(model.num_params()),
            )
            writer.set_final(**final)
            mlf.log_metrics({k: v for k, v in final.items()})

            log.info(
                "dynamic_grokking done | dynamic_success=%.3f static_success=%.3f (diff=%.3f) over %d problems",
                dynamic_success, static_success, dynamic_success - static_success, len(hard_problems),
            )

            notes = (
                "Toy dynamic-evaluation comparison at equal compute. A non-positive "
                "dynamic_minus_static is a legitimate, honestly-recorded outcome at this scale."
            )
            out = writer.finish(status="completed", notes=notes)
        except Exception as exc:  # noqa: BLE001
            writer.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    log.info("wrote %s", out)
    return out
