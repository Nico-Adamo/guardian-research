"""Train a tiny transformer on synthetic arithmetic with a chosen schedule.

This is the catapult-arithmetic lab's training entry point. One loop runs any
schedule (the schedule sets lr/wd each step). It logs everything the hypothesis
needs: train/val loss, easy & hard accuracy, the memorization gap, gradient
norm, lr, wd, and checkpoint metadata.

Runs on CPU in seconds at the toy scale; the same code scales up via config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ...common.artifacts import RunWriter, new_run_id
from ...common.logging import get_logger
from ...common.seeding import seed_everything
from ...data.arithmetic import ArithmeticConfig, build_splits
from ...models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from ...schedules.schedules import ScheduleConfig, get_schedule
from ...tracking.mlflow_client import start_run

log = get_logger(__name__)


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


@torch.no_grad()
def _eval_loss(model: TinyTransformer, loader: DataLoader, device: str, max_batches: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (x, y, m) in enumerate(loader):
        if i >= max_batches:
            break
        x, y, m = x.to(device), y.to(device), m.to(device)
        total += float(_masked_ce(model(x), y, m).item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def _eval_accuracy(model, dataset, tok, n: int, max_new: int) -> float:
    """Exact-match accuracy via greedy autoregressive decoding."""
    model.eval()
    n = min(n, len(dataset))
    correct = 0
    for idx in range(n):
        prompt, answer = dataset.text(idx)
        prompt_ids = torch.tensor([tok.bos_id] + tok.encode(prompt), dtype=torch.long)
        gen = model.generate_greedy(prompt_ids, max_new_tokens=max_new, eos_id=tok.eos_id)
        if tok.decode(gen) == answer:
            correct += 1
    return correct / max(1, n)


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item()) ** 2
    return total**0.5


def run(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = _resolve_device(str(cfg.get("device", "auto")))
    tr = cfg.get("train", {})
    data_cfg_dict = cfg.get("data", {})
    model_cfg_dict = cfg.get("model", {})
    sched_dict = cfg.get("schedule", {})

    # -- data ------------------------------------------------------------- #
    acfg = ArithmeticConfig(
        **{k: v for k, v in data_cfg_dict.items() if k in ArithmeticConfig.__dataclass_fields__},
        seed=seed,
    )
    splits = build_splits(acfg)
    tok = splits["tokenizer"]
    max_len = splits["max_len"]
    train_ds, easy_ds, hard_ds = splits["train"], splits["easy_eval"], splits["hard_eval"]

    batch_size = int(tr.get("batch_size", 256))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    easy_loader = DataLoader(easy_ds, batch_size=batch_size)
    hard_loader = DataLoader(hard_ds, batch_size=batch_size)

    # -- model ------------------------------------------------------------ #
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

    # -- optimizer + schedule -------------------------------------------- #
    base_lr = float(tr.get("lr", 1e-3))
    base_wd = float(tr.get("weight_decay", 0.01))
    scfg = ScheduleConfig.from_dict({**sched_dict, "base_lr": base_lr, "base_wd": base_wd})
    schedule = get_schedule(scfg)
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_wd, betas=(0.9, 0.98))

    max_steps = int(tr.get("max_steps", 800))
    eval_every = int(tr.get("eval_every", 200))
    eval_n = int(tr.get("eval_n", 128))
    eval_batches = int(tr.get("eval_batches", 2))
    log_every = int(tr.get("log_every", 50))
    grad_clip = float(tr.get("grad_clip", 1.0))
    max_answer_len = max_len  # safe upper bound for generation

    run_id = new_run_id(prefix=str(sched_dict.get("name", "run")))
    experiment = str(cfg.get("experiment", "arithmetic_catapult"))

    writer = RunWriter(experiment=experiment, run_id=run_id, seed=seed)
    writer.set_config(cfg)
    writer.set_params(
        schedule=sched_dict.get("name"),
        model=model_cfg_dict.get("name", "tiny_transformer"),
        num_params=model.num_params(),
        base_lr=base_lr,
        base_wd=base_wd,
        max_steps=max_steps,
        batch_size=batch_size,
        op=acfg.op,
        train_digits=f"{acfg.train_min_digits}-{acfg.train_max_digits}",
        hard_digits=f"{acfg.hard_min_digits}-{acfg.hard_max_digits}",
        device=device,
        **{f"data_{k}": v for k, v in splits["meta"].items()},
    )

    log.info(
        "arithmetic: %s | schedule=%s seed=%d params=%d device=%s steps=%d",
        experiment, scfg.name, seed, model.num_params(), device, max_steps,
    )

    with start_run(experiment, run_id, tags={"schedule": scfg.name, "seed": str(seed)}) as mlf:
        mlf.log_params({**writer.result.params})
        step = 0
        data_iter = iter(train_loader)
        try:
            while step < max_steps:
                try:
                    x, y, m = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y, m = next(data_iter)
                x, y, m = x.to(device), y.to(device), m.to(device)

                lr, wd = schedule(step, max_steps)
                for g in optim.param_groups:
                    g["lr"] = lr
                    g["weight_decay"] = wd

                model.train()
                optim.zero_grad(set_to_none=True)
                loss = _masked_ce(model(x), y, m)
                loss.backward()
                gnorm = _grad_norm(model)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()

                if step % log_every == 0:
                    writer.log_metrics(step, train_loss=float(loss.item()), lr=lr, wd=wd, grad_norm=gnorm)
                    mlf.log_metrics({"train_loss": float(loss.item()), "lr": lr, "wd": wd, "grad_norm": gnorm}, step=step)

                if step % eval_every == 0 or step == max_steps - 1:
                    easy_loss = _eval_loss(model, easy_loader, device, eval_batches)
                    hard_loss = _eval_loss(model, hard_loader, device, eval_batches)
                    train_acc = _eval_accuracy(model, train_ds, tok, eval_n, max_answer_len)
                    easy_acc = _eval_accuracy(model, easy_ds, tok, eval_n, max_answer_len)
                    hard_acc = _eval_accuracy(model, hard_ds, tok, eval_n, max_answer_len)
                    mem_gap = train_acc - easy_acc
                    writer.log_metrics(
                        step,
                        easy_loss=easy_loss,
                        hard_loss=hard_loss,
                        train_acc=train_acc,
                        easy_acc=easy_acc,
                        hard_acc=hard_acc,
                        memorization_gap=mem_gap,
                    )
                    mlf.log_metrics(
                        {
                            "easy_loss": easy_loss,
                            "hard_loss": hard_loss,
                            "train_acc": train_acc,
                            "easy_acc": easy_acc,
                            "hard_acc": hard_acc,
                            "memorization_gap": mem_gap,
                        },
                        step=step,
                    )
                    log.info(
                        "step %d | train_loss=%.4f easy_acc=%.3f hard_acc=%.3f mem_gap=%.3f lr=%.2e wd=%.2e",
                        step, float(loss.item()), easy_acc, hard_acc, mem_gap, lr, wd,
                    )
                step += 1

            # -- final eval + checkpoint -------------------------------- #
            easy_acc = _eval_accuracy(model, easy_ds, tok, eval_n, max_answer_len)
            hard_acc = _eval_accuracy(model, hard_ds, tok, eval_n, max_answer_len)
            train_acc = _eval_accuracy(model, train_ds, tok, eval_n, max_answer_len)
            writer.set_final(
                final_train_loss=float(loss.item()),
                final_easy_acc=easy_acc,
                final_hard_acc=hard_acc,
                final_train_acc=train_acc,
                final_memorization_gap=train_acc - easy_acc,
                num_params=model.num_params(),
            )

            if bool(tr.get("checkpoint", True)):
                ckpt = writer.add_artifact("checkpoint_final.pt")
                torch.save({"model": model.state_dict(), "config": mcfg.__dict__}, ckpt)
                writer.set_params(
                    checkpoint_path=str(ckpt.relative_to(writer.dir)),
                    checkpoint_bytes=ckpt.stat().st_size,
                    checkpoint_step=max_steps,
                )
                mlf.log_artifact(str(ckpt))

            out = writer.finish(status="completed")
        except Exception as exc:  # noqa: BLE001
            writer.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise

    log.info("wrote %s", out)
    return out
