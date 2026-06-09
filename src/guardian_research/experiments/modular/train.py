"""Train a tiny transformer on modular arithmetic and watch for grokking.

Logs the two curves that matter: train_acc (memorization) and val_acc (the
generalizing algorithm). The grokking signal is val_acc rising *long after*
train_acc saturates. Schedule/weight-decay are the experimental knob (H001) —
cyclic-WD in particular is the artificial "sleep" that should help the model
escape the memorization basin.

Single loop, any schedule (sets lr/wd per step). CPU-friendly: 4-token
sequences, a small classifier head on the final position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...common.artifacts import RunWriter, new_run_id
from ...common.logging import get_logger
from ...common.seeding import seed_everything
from ...data.modular import ModularConfig, build_modular_splits
from ...models.tiny_transformer import TinyTransformer, TinyTransformerConfig
from ...schedules.schedules import ScheduleConfig, get_schedule
from ...tracking.mlflow_client import start_run

log = get_logger(__name__)


def _resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _accuracy(model: TinyTransformer, X: torch.Tensor, y: torch.Tensor, device: str, batch: int = 2048) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), batch):
        xb = X[i : i + batch].to(device)
        logits = model(xb)[:, -1, :]  # predict at the final (eq) position
        pred = logits.argmax(dim=-1).cpu()
        correct += int((pred == y[i : i + batch]).sum().item())
    return correct / max(1, len(X))


def run(cfg: dict[str, Any]) -> Path:
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = _resolve_device(str(cfg.get("device", "auto")))
    tr = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    sched_dict = cfg.get("schedule", {})

    mcfg_data = ModularConfig(
        p=int(data_cfg.get("p", 97)),
        op=str(data_cfg.get("op", "+")),
        train_frac=float(data_cfg.get("train_frac", 0.4)),
        seed=seed,
    )
    splits = build_modular_splits(mcfg_data)
    Xtr, ytr = splits["X_train"], splits["y_train"]
    Xval, yval = splits["X_val"], splits["y_val"]
    vocab_size = splits["vocab_size"]

    model = TinyTransformer(
        TinyTransformerConfig(
            vocab_size=vocab_size,
            max_len=splits["seq_len"],
            d_model=int(model_cfg.get("d_model", 128)),
            n_layers=int(model_cfg.get("n_layers", 2)),
            n_heads=int(model_cfg.get("n_heads", 4)),
            d_ff=int(model_cfg.get("d_ff", 512)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            tie_weights=bool(model_cfg.get("tie_weights", True)),
            pos_encoding=str(model_cfg.get("pos_encoding", "learned")),
        )
    ).to(device)

    base_lr = float(tr.get("lr", 1e-3))
    base_wd = float(tr.get("weight_decay", 1.0))  # heavy WD is what makes grokking happen
    scfg = ScheduleConfig.from_dict({**sched_dict, "base_lr": base_lr, "base_wd": base_wd})
    schedule = get_schedule(scfg)
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=base_wd, betas=(0.9, 0.98))

    max_steps = int(tr.get("max_steps", 4000))
    batch_size = int(tr.get("batch_size", 512))
    eval_every = int(tr.get("eval_every", 200))
    log_every = int(tr.get("log_every", 500))
    grok_threshold = float(tr.get("grok_threshold", 0.9))

    experiment = str(cfg.get("experiment", "arithmetic_modular_grok"))
    run_id = new_run_id(prefix=str(sched_dict.get("name", "modgrok")))
    writer = RunWriter(experiment=experiment, run_id=run_id, seed=seed)
    writer.set_config(cfg)
    writer.set_params(
        schedule=sched_dict.get("name"),
        task=f"({mcfg_data.op} mod {mcfg_data.p})",
        train_frac=mcfg_data.train_frac,
        num_params=model.num_params(),
        base_lr=base_lr,
        base_wd=base_wd,
        max_steps=max_steps,
        device=device,
        **{f"data_{k}": v for k, v in splits["meta"].items()},
    )

    log.info(
        "modular grokking: p=%d op=%s frac=%.2f | schedule=%s wd=%.3g steps=%d device=%s",
        mcfg_data.p, mcfg_data.op, mcfg_data.train_frac, scfg.name, base_wd, max_steps, device,
    )

    grok_step = -1
    n = len(Xtr)
    with start_run(experiment, run_id, tags={"schedule": scfg.name, "seed": str(seed)}) as mlf:
        mlf.log_params({**writer.result.params})
        g = torch.Generator().manual_seed(seed)
        for step in range(max_steps):
            lr, wd = schedule(step, max_steps)
            for grp in optim.param_groups:
                grp["lr"] = lr
                grp["weight_decay"] = wd

            idx = torch.randint(0, n, (min(batch_size, n),), generator=g)
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            model.train()
            optim.zero_grad(set_to_none=True)
            logits = model(xb)[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optim.step()

            if step % log_every == 0:
                writer.log_metrics(step, train_loss=float(loss.item()), lr=lr, wd=wd)
                mlf.log_metrics({"train_loss": float(loss.item()), "lr": lr, "wd": wd}, step=step)

            if step % eval_every == 0 or step == max_steps - 1:
                train_acc = _accuracy(model, Xtr, ytr, device)
                val_acc = _accuracy(model, Xval, yval, device)
                grok_gap = train_acc - val_acc
                if grok_step < 0 and val_acc >= grok_threshold:
                    grok_step = step
                writer.log_metrics(step, train_acc=train_acc, val_acc=val_acc, grok_gap=grok_gap)
                mlf.log_metrics({"train_acc": train_acc, "val_acc": val_acc, "grok_gap": grok_gap}, step=step)
                if step % log_every == 0 or step == max_steps - 1:
                    log.info("step %d | train_acc=%.3f val_acc=%.3f gap=%.3f wd=%.2e",
                             step, train_acc, val_acc, grok_gap, wd)

        final_train = _accuracy(model, Xtr, ytr, device)
        final_val = _accuracy(model, Xval, yval, device)
        writer.set_final(
            final_train_acc=final_train,
            final_val_acc=final_val,
            final_grok_gap=final_train - final_val,
            grok_step=float(grok_step),  # -1 means "never grokked within budget"
            grokked=1.0 if grok_step >= 0 else 0.0,
        )
        out = writer.finish(status="completed")

    log.info("wrote %s (grokked=%s at step %s)", out, grok_step >= 0, grok_step)
    return out
