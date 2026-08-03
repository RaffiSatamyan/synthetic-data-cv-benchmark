# Synthetic Data for Computer Vision: A Reproducible Benchmark

This repository supports a benchmark paper studying **when synthetic images improve downstream computer-vision performance** across classification, object detection, and semantic segmentation.

The benchmark varies:

- downstream task and dataset;
- amount of available real training data;
- synthetic-data generation method;
- synthetic-to-real data ratio;
- random seed.

## Core research questions

1. How does the value of synthetic data change as real-data availability changes?
2. Which generation families transfer best to classification, detection, and segmentation?
3. How much synthetic data is useful before performance saturates or decreases?
4. Do image-quality metrics predict downstream utility?
5. How robust are conclusions across datasets, architectures, and random seeds?

## Repository layout

```text
configs/                 Experiment definitions
  generators/            Generator-family configurations
  tasks/                 Task and dataset configurations
data/README.md            Dataset placement and preparation rules
docs/                     Experimental protocol and reporting rules
paper/                    Tables, figures, and manuscript-facing assets
results/                  Result schema and aggregated outputs
scripts/                  Command-line entry points
src/synthbench/           Grid construction and experiment utilities
tests/                    Reproducibility checks
.github/workflows/        Continuous integration
```

## Experiment counts

The repository intentionally defines two grids:

- **Pilot grid: 18 runs** — one representative dataset per task, two real-data fractions, and three regimes.
- **Full grid: 864 runs** — six task/dataset combinations, four real-data fractions, three seeds, three non-synthetic baselines/augmentation regimes, and nine synthetic regimes.

The full grid is deduplicated: real-only and classical-augmentation baselines are not repeated for every synthetic ratio or generator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Build experiment manifests

```bash
python scripts/build_grid.py --preset pilot --output manifests/pilot.csv
python scripts/build_grid.py --preset full --output manifests/full.csv
```

Expected output:

```text
pilot: 18 experiments
full: 864 experiments
```

## Run one experiment

```bash
python scripts/run_experiment.py \
  --manifest manifests/pilot.csv \
  --index 0 \
  --output-dir outputs
```

`run_experiment.py` currently provides a strict, reproducible interface and a placeholder runner. Task-specific training and generator adapters should implement the interfaces described in `docs/experimental_protocol.md`.

## Aggregate results

```bash
python scripts/aggregate_results.py \
  --input-glob "outputs/**/metrics.json" \
  --output results/summary.csv
```

## Two-student division

A practical split is:

- **Student A — synthetic-data pipeline:** generator adapters, prompt/conditioning protocol, generation cost, filtering, and image-quality/diversity analysis.
- **Student B — downstream evaluation:** task models, real-data subsampling, training, metrics, statistical tests, and result aggregation.

Both students should share the experiment manifest, dataset splits, result schema, and code review.

## Reproducibility rules

- Never commit raw datasets, generated images, model checkpoints, API keys, or W&B credentials.
- Freeze train/validation/test splits before the first benchmark run.
- Use at least three seeds for paper conclusions.
- Report mean, standard deviation, paired confidence intervals, and effect sizes.
- Log failed or excluded runs rather than silently deleting them.
- Keep generation and downstream-compute costs in the final comparison.

See `docs/experimental_protocol.md` for the full protocol.