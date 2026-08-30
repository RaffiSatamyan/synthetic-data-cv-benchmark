"""Friendly single-run entry point: hand it data, a model, and params.

This mirrors ``synthbench.training.train`` on the command line -- provision the
dataset if needed, train from scratch (or reuse an exact existing checkpoint),
evaluate on the test split, and write figures.

Example:
    python scripts/run.py --dataset imagenet100 --model resnet18 \
        --real-fraction 0.1 --generator classical_augmentation --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthbench.training import train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate one run")
    parser.add_argument("--dataset", required=True, help="Dataset config name, e.g. imagenet100")
    parser.add_argument("--model", required=True, help="Model config name, e.g. resnet18")
    parser.add_argument("--generator", default="real_only")
    parser.add_argument("--real-fraction", type=float, default=1.0)
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override the recipe's epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--generation-id", default=None)
    parser.add_argument("--force", action="store_true", help="Retrain even if a checkpoint exists")
    parser.add_argument("--no-provision", action="store_true", help="Assume data is already local")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    training_overrides: dict[str, int] = {}
    if args.epochs is not None:
        training_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size

    run_dir = train(
        args.dataset,
        args.model,
        generator=args.generator,
        real_fraction=args.real_fraction,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
        configs_dir=args.configs_dir,
        outputs_dir=args.outputs,
        generation_id=args.generation_id,
        training_overrides=training_overrides or None,
        force=args.force,
        provision=not args.no_provision,
        evaluate=not args.no_eval,
        make_plots=not args.no_plots,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
