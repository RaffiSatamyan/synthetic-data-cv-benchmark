# Data directory

Create one folder per dataset:

```text
data/<dataset>/
├── raw/
├── splits/
│   ├── full/train.csv
│   ├── 1pct/train.csv
│   ├── 3pct/train.csv
│   ├── 10pct/train.csv
│   ├── val.csv
│   └── test.csv
└── synthetic/<generator>/<generation_id>/
```

Raw and generated images are ignored by Git. Split CSVs, generation configs, and small provenance manifests may be committed when licensing and privacy permit.
