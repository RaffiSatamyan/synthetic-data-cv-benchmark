# Contributing

Use small branches and keep responsibilities separate:

- generator work: `generation.py`, `annotation.py`, `filtering.py`, and generator configs;
- downstream work: `datasets.py`, `models.py`, `training.py`, `evaluation.py`, and model configs;
- shared work: experiment manifests, split files, result schemas, and README.

Before opening a pull request:

```bash
pytest -q
ruff check .
```

Do not commit datasets, generated images, checkpoints, API keys, or private W&B files. Commit split CSV files, configs, manifests, tests, and aggregated paper results.
