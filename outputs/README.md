# Run outputs

Each run writes to `outputs/<run_id>/` and should contain:

- `config.yaml`: fully resolved configuration;
- `environment.json`: package/GPU environment;
- `status.json`: created/running/completed/failed;
- `best.pt`: best validation checkpoint;
- `last.pt`: latest resumable checkpoint;
- `metrics.json`: final validation/test metrics;
- `predictions.csv`: optional per-sample outputs;
- `train.log`: human-readable logs.

The run directory contents are ignored by Git.
