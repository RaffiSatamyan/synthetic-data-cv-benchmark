# Checkpoints

- `generators/`: reusable locally trained generator weights, LoRAs, or GAN checkpoints.
- `teachers/`: frozen detector/segmenter weights used to label synthetic data.

Downstream experiment checkpoints do not belong here. They are written to `outputs/<run_id>/best.pt` and `last.pt`.

Large weight files are ignored by Git. Record their source, exact revision, and checksum in the corresponding YAML config.
