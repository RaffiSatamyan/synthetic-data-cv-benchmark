"""Local / server entry point: download the data, then train — one command.

Unlike the Kaggle notebook flow (where the data is already mounted), a fresh
server has nothing on disk. This script makes the data appear and trains in one
go: it provisions the dataset (download + extract into the right directory so
run.py-style path resolution just works), then trains from scratch — optionally
sweeping every data-size subset.

Examples:
    # quick single run on Imagenette (downloads ~326 MB the first time)
    python scripts/run_server.py --dataset imagenette --model resnet18 --epochs 10

    # full data-size sweep (1% -> 100%), each trained from scratch, then the
    # accuracy-vs-data-size figure
    python scripts/run_server.py --dataset imagenette --model resnet18 --all-subsets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from synthbench.config import load_yaml
from synthbench.provision import ensure_dataset
from synthbench.training import train

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download data + train on a local/server box")
    parser.add_argument("--dataset", default="imagenette", help="Dataset config name")
    parser.add_argument("--model", default="resnet18", help="Model config name")
    parser.add_argument("--generator", default="real_only")
    parser.add_argument("--real-fraction", type=float, default=1.0)
    parser.add_argument("--synthetic-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override the recipe's epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Sweep every real_fraction (1%%->100%%), each from scratch, then plot.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Where to download/extract the images (sets the dataset's image root).",
    )
    parser.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    # Optional override of where the images land, honoured via the dataset's
    # image_root env interpolation (e.g. IMAGENETTE_ROOT).
    if args.data_dir is not None:
        os.environ[f"{args.dataset.upper()}_ROOT"] = str(args.data_dir)

    # 1) Make the data present (download + extract + build the split). Explicit,
    #    so the download step is visible rather than hidden inside training.
    dataset_cfg = load_yaml(args.configs_dir / "datasets" / f"{args.dataset}.yaml")
    print(f"[1/2] Provisioning '{args.dataset}' into {dataset_cfg['data_root']} ...")
    ensure_dataset(dataset_cfg)
    print("      data ready.")

    # 2) Train (data already provisioned above).
    training_overrides: dict[str, int] = {}
    if args.epochs is not None:
        training_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size

    print(f"[2/2] Training {args.model} on {args.dataset}"
          f"{' (all subsets)' if args.all_subsets else ''} ...")
    result = train(
        args.dataset,
        args.model,
        generator=args.generator,
        real_fraction=args.real_fraction,
        synthetic_ratio=args.synthetic_ratio,
        seed=args.seed,
        configs_dir=args.configs_dir,
        outputs_dir=args.outputs,
        training_overrides=training_overrides or None,
        force=args.force,
        provision=False,  # already provisioned in step 1
        evaluate=not args.no_eval,
        make_plots=not args.no_plots,
        all_subsets=args.all_subsets,
    )
    if isinstance(result, list):
        print(f"Sweep complete: {len(result)} subset runs")
        for path in result:
            print(f"  {path}")
    else:
        print(f"Run complete: {result}")


if __name__ == "__main__":
    main()
