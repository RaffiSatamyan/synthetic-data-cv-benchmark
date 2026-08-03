# Synthetic Data CV Benchmark

A compact, reproducible research repository for studying when synthetic images improve computer-vision models.

The benchmark compares real data only, classical augmentation, and multiple synthetic-image generators across different real-data sizes, synthetic-to-real ratios, seeds, and computer-vision tasks.

This repository is intentionally small. Each file has one clear responsibility, while datasets, generated images, and checkpoints stay outside Git.

## Repository structure

```text
.
├── configs/                 Reusable dataset, generator, model, and training settings
├── experiments/             Experiment grids: which combinations should be executed
├── src/synthbench/          Reusable Python implementation
├── scripts/                 Thin command-line entry points
├── data/                    Dataset-centered real splits and synthetic datasets
├── checkpoints/             Reusable generator and teacher checkpoints
├── outputs/                 One directory per downstream training run
├── results/                 Aggregated paper results, figures, and tables
├── tests/                   Fast correctness and smoke tests
└── .github/workflows/       Continuous integration
```

## Data organization

Every dataset should use this layout:

```text
data/<dataset_name>/
├── raw/                     Original real data; treat as read-only
├── splits/
│   ├── full/train.csv       Full real training split
│   ├── 1pct/train.csv       Fixed 1% subset
│   ├── 3pct/train.csv       Fixed 3% subset
│   ├── 10pct/train.csv      Fixed 10% subset
│   ├── val.csv              Validation set for model/checkpoint selection
│   └── test.csv             Untouched real test set
└── synthetic/
    └── <generator>/
        └── <generation_id>/
            ├── images/      Generated images
            ├── annotations/ Boxes or masks when needed
            ├── manifest.csv One row per generated sample
            └── config.yaml  Exact generation settings
```

The percentage folders contain CSV manifests, not copied images. Whenever possible, use nested subsets:

```text
1% subset ⊂ 3% subset ⊂ 10% subset ⊂ full training set
```

## Responsibilities

### `configs/`

Reusable component settings. These files describe one dataset, generator, model, or the shared training procedure. They contain settings, not training logic.

- `configs/datasets/*.yaml`: task, paths, number of classes, split locations, and primary metric.
- `configs/generators/*.yaml`: generator name, exact model revision/checkpoint, prompts, and inference parameters.
- `configs/models/*.yaml`: downstream architecture and initialization.
- `configs/training.yaml`: epochs, optimizer, batch size, checkpointing, logging, and seed settings.

### `experiments/`

Defines which combinations should run.

- `pilot.yaml`: small 18-run grid for testing the full pipeline.
- `full.yaml`: larger 288-run benchmark grid.
- `paper_runs.csv`: frozen list of runs included in the final paper. Do not silently regenerate it after results are finalized.

### `src/synthbench/`

Reusable implementation:

- `config.py`: reads and validates YAML and builds resolved run configurations.
- `datasets.py`: loads manifests, checks data leakage, creates nested subsets, and mixes real/synthetic samples.
- `generation.py`: common generator interface, resumable generation loop, image saving, and provenance manifests.
- `annotation.py`: assigns classification labels or creates teacher boxes/masks.
- `filtering.py`: marks corrupt, duplicate, empty, or low-confidence synthetic samples and records reasons.
- `models.py`: builds downstream models. Model weights do not belong in this file.
- `training.py`: run directories, checkpoint format, resume logic, and task-training entry point.
- `evaluation.py`: task metrics and validation-selected checkpoint evaluation.
- `experiments.py`: deduplicated grids, deterministic run IDs, and CSV manifests.
- `utils.py`: seeds, hashes, atomic writes, and environment metadata.

### `scripts/`

Thin command-line wrappers. Reusable logic belongs in `src/synthbench/`.

- `prepare_data.py`: validates a source manifest and creates full/1%/3%/10%/validation/test manifests.
- `generate.py`: generates one synthetic dataset and writes its provenance manifest.
- `train.py`: resolves one experiment row and starts downstream training.
- `evaluate.py`: evaluates one completed run using `best.pt`.
- `aggregate.py`: combines `outputs/*/metrics.json` into paper-level CSV files.

### `checkpoints/`

Reusable weights only:

- `generators/`: locally trained GANs, diffusion LoRAs, or other generator weights.
- `teachers/`: frozen detector/segmenter weights used to annotate synthetic data.

Downstream run checkpoints belong in `outputs/<run_id>/`, not here.

### `outputs/`

Each run writes:

```text
outputs/<run_id>/
├── config.yaml              Fully resolved configuration actually used
├── environment.json         Python, PyTorch, CUDA, and GPU information
├── status.json              created/running/completed/failed
├── best.pt                  Best validation checkpoint
├── last.pt                  Latest checkpoint for resuming
├── metrics.json             Final validation/test metrics and resource usage
├── predictions.csv          Optional sample-level predictions
└── train.log                Human-readable log
```

`best.pt` must be selected only with validation performance. Test performance must never choose checkpoints or hyperparameters.

### `results/`

Paper-level outputs:

- `all_runs.csv`: one row per completed run.
- `summary.csv`: means and standard deviations across seeds.
- `figures/`: plots produced from aggregated results.
- `tables/`: CSV or LaTeX tables produced from aggregated results.

Do not manually copy final values from W&B into the paper.

## Synthetic manifest

Each generation should have a `manifest.csv` with at least:

```text
sample_id,image_path,annotation_path,label,generator,prompt,generation_seed,accepted,rejection_reason
```

Also record the generator revision/checkpoint hash, teacher checkpoint hash, confidence, and image SHA-256 when available.

## Run identifiers

Runs receive deterministic IDs such as:

```text
cifar10-resnet18-stable-diffusion-rf010-sr100-seed0-a81f29c3
```

The suffix hashes the resolved run settings, so different experiments cannot silently overwrite the same output directory.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For training and generation:

```bash
pip install -e ".[training,generation,tracking,dev]"
```

## Build experiment manifests

```bash
python -m synthbench.experiments \
  --experiment experiments/pilot.yaml \
  --output experiments/pilot_runs.csv

python -m synthbench.experiments \
  --experiment experiments/full.yaml \
  --output experiments/full_runs.csv
```

## Typical workflow

```bash
# 1. Create fixed real-data splits
python scripts/prepare_data.py \
  --source-manifest path/to/all_real_samples.csv \
  --dataset-dir data/cifar10 \
  --label-column label \
  --seed 42

# 2. Test generation storage with the placeholder adapter
python scripts/generate.py \
  --dataset-config configs/datasets/cifar10.yaml \
  --generator-config configs/generators/stable_diffusion.yaml \
  --generation-id smoke_v1 \
  --placeholder

# 3. Resolve one pilot experiment
python scripts/train.py \
  --experiment experiments/pilot.yaml \
  --index 0 \
  --dry-run

# 4. Aggregate completed run metrics
python scripts/aggregate.py --outputs outputs --results results
```

Generator adapters and complete task-specific training loops are deliberate extension points. The interfaces, paths, manifests, checkpoint format, and experiment system are prepared so they can be implemented without reorganizing the repository.

## Checkpoint rules

A downstream checkpoint should contain:

```text
model_state_dict
optimizer_state_dict
scheduler_state_dict
epoch
global_step
best_validation_metric
resolved_config
config_hash
random_states
```

Generator checkpoints and downstream checkpoints must remain separate.

## Research rules

1. Never use test data for generator training, prompts, filtering, checkpoint selection, or hyperparameter tuning.
2. Keep exact, versioned split CSV files.
3. Record provenance for every synthetic image.
4. Save the resolved configuration for every run.
5. Use validation metrics to select `best.pt`.
6. Keep failed runs in the manifest with a failure reason.
7. Save metrics locally even when W&B is enabled.
8. Do not commit raw images, generated images, or model weights.
9. Aggregate across multiple seeds and report uncertainty.
10. Generate paper tables and figures automatically from result CSV files.
