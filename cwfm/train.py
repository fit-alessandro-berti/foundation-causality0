from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import TASK_REGIME, collate_episodes, generate_episode
from .model import CWFM, CWFMConfig


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    scale = output["effect_scale"].clamp(0.04, 4.0)
    effect = 0.5 * (((target - output["effect_mean"]) / scale) ** 2 + 2 * scale.log())
    estimable = (batch["identified"] * batch["supported"]).bool()
    effect_loss = effect[estimable].mean()
    point_loss = F.mse_loss(
        output["effect_mean"][estimable], target[estimable]
    )
    id_loss = F.binary_cross_entropy_with_logits(
        output["identification_logit"], batch["identified"]
    )
    support_loss = F.binary_cross_entropy_with_logits(
        output["support_logit"], batch["supported"]
    )
    mechanism_loss = F.cross_entropy(output["mechanism_logits"], batch["mechanism"])
    route_loss = F.binary_cross_entropy_with_logits(
        output["flexible_route_logit"], batch["route_flexible"]
    )
    regime = batch["task"] == TASK_REGIME
    structure_loss = (
        F.cross_entropy(
            output["structure_logits"][regime], batch["structure_target"][regime]
        )
        if regime.any()
        else target.new_zeros(())
    )
    variable_mask = batch["variable_mask"]
    graph_mask = variable_mask[:, :, None] & variable_mask[:, None, :]
    eye = torch.eye(graph_mask.shape[-1], device=graph_mask.device, dtype=torch.bool)
    graph_mask &= ~eye.unsqueeze(0)
    graph_loss = F.binary_cross_entropy_with_logits(
        output["graph_logits"][graph_mask],
        batch["graph"][graph_mask],
        pos_weight=target.new_tensor(2.0),
    )
    world_diversity = -output["particle_means"].std(-1).mean()
    loss = (
        effect_loss
        + point_loss
        + 0.7 * id_loss
        + 0.7 * support_loss
        + 0.25 * mechanism_loss
        + 0.25 * route_loss
        + 1.5 * structure_loss
        + 0.15 * graph_loss
        + 0.01 * world_diversity
    )
    parts = {
        "loss": float(loss.detach()),
        "effect": float(effect_loss.detach()),
        "identification": float(id_loss.detach()),
        "support": float(support_loss.detach()),
        "mechanism": float(mechanism_loss.detach()),
        "structure": float(structure_loss.detach()),
        "graph": float(graph_loss.detach()),
    }
    return loss, parts


@torch.no_grad()
def validate(model: CWFM, device: torch.device, seed: int, batches: int = 12) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[float]] = {}
    for index in range(batches):
        episodes = [
            generate_episode(seed + index * 100 + offset)
            for offset in range(16)
        ]
        batch = _to_device(collate_episodes(episodes, model.config.max_variables), device)
        output = model(batch)
        _, parts = compute_loss(output, batch)
        for key, value in parts.items():
            totals.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in totals.items()}


def train(
    output: Path,
    steps: int = 1400,
    batch_size: int = 16,
    seed: int = 1729,
    device_name: str = "auto",
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        ("cpu" if device_name == "auto" else device_name)
    )
    config = CWFMConfig()
    model = CWFM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    history: list[dict[str, float | int]] = []
    started = time.time()
    model.train()
    for step in range(steps):
        episodes = [
            generate_episode(seed + step * batch_size + offset)
            for offset in range(batch_size)
        ]
        batch = _to_device(collate_episodes(episodes, config.max_variables), device)
        optimizer.zero_grad(set_to_none=True)
        output_values = model(batch)
        loss, parts = compute_loss(output_values, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == steps - 1:
            record = {
                "step": step + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **parts,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    validation = validate(model, device, seed + 10_000_000)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "validation": validation,
        "runtime_seconds": time.time() - started,
        "torch_version": torch.__version__,
    }
    torch.save(payload, output)
    history_path = output.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "seed": seed,
                    "steps": steps,
                    "batch_size": batch_size,
                    "device": str(device),
                    **config.to_dict(),
                },
                "history": history,
                "validation": validation,
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        )
        + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the CWFM PyTorch prototype")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cwfm/checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args.output, args.steps, args.batch_size, args.seed, args.device)


if __name__ == "__main__":
    main()
