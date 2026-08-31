"""Make a dataset present on whatever machine the run lands on.

The training loop assumes ``data_root`` already holds split manifests. On a
fresh remote box (a rented 5090, a Kaggle/Colab runner) nothing is there yet,
so this module bridges "hosted on Kaggle" -> "local files + manifests":

1. ``download_kaggle_dataset`` pulls a Kaggle dataset into a local cache.
2. ``build_image_manifest`` turns an ImageFolder-style tree
   (``<root>/<class>/<image>``) into a ``sample_id, image_path, label`` frame,
   which is exactly the source manifest ``prepare_data.py`` consumes.
3. ``ensure_dataset`` ties it together from a dataset config's ``provision``
   block and is safe to call on every run (it no-ops once splits exist).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def select_classes(
    images_root: str | Path,
    *,
    include_classes: list[str] | set[str] | None = None,
    max_classes: int | None = None,
) -> list[Path]:
    """Return the class subdirectories to scan, deterministically ordered.

    ``include_classes`` restricts to an explicit set of class names (e.g. a
    100-wnid ImageNet-100 list); ``max_classes`` otherwise takes the first N
    sorted class folders. With neither, every class folder is used. Sorting keeps
    the selection identical on every machine, which is what makes the downstream
    subsets reproducible.
    """
    images_root = Path(images_root)
    if not images_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {images_root}")
    class_dirs = sorted(p for p in images_root.iterdir() if p.is_dir())

    if include_classes is not None:
        wanted = {str(name) for name in include_classes}
        chosen = [p for p in class_dirs if p.name in wanted]
        missing = wanted - {p.name for p in chosen}
        if missing:
            raise ValueError(
                f"{len(missing)} requested classes are absent under {images_root}: "
                f"{sorted(missing)[:5]}..."
            )
        return chosen
    if max_classes is not None:
        return class_dirs[: int(max_classes)]
    return class_dirs


def build_image_manifest(
    images_root: str | Path,
    *,
    path_relative_to: str | Path | None = None,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    split_prefix: str | None = None,
    include_classes: list[str] | set[str] | None = None,
    max_classes: int | None = None,
    absolute_paths: bool = False,
) -> pd.DataFrame:
    """Scan ``<images_root>/<class>/<image>`` into a source manifest frame.

    ``image_path`` is stored relative to ``path_relative_to`` (default
    ``images_root``) so manifests stay portable -- unless ``absolute_paths`` is
    set, in which case absolute paths are stored (needed when images live on a
    read-only mount outside the writable data root, e.g. Kaggle's
    ``/kaggle/input``). ``include_classes`` / ``max_classes`` subset the classes
    (turning ImageNet-1k into ImageNet-100). ``sample_id`` is namespaced by
    ``split_prefix`` to keep ids disjoint when several sources feed one manifest.
    """
    images_root = Path(images_root)
    base = Path(path_relative_to) if path_relative_to is not None else images_root
    suffixes = {ext.lower() for ext in extensions}
    class_dirs = select_classes(
        images_root, include_classes=include_classes, max_classes=max_classes
    )

    records: list[dict[str, str]] = []
    for class_dir in class_dirs:
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in suffixes:
                continue
            stem = image_path.relative_to(images_root).as_posix().replace("/", "_")
            sample_id = f"{split_prefix}_{stem}" if split_prefix else stem
            if absolute_paths:
                stored_path = str(image_path.resolve())
            else:
                stored_path = image_path.relative_to(base).as_posix()
            records.append(
                {"sample_id": sample_id, "image_path": stored_path, "label": label}
            )

    if not records:
        raise ValueError(
            f"No images with extensions {sorted(suffixes)} found under {images_root}"
        )
    frame = pd.DataFrame.from_records(records)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Manifest has duplicate sample_id values; check split_prefix")
    return frame


def write_source_manifest(
    frame: pd.DataFrame, destination: str | Path
) -> Path:
    """Persist a manifest frame, creating parent directories as needed."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def download_and_extract(
    url: str, destination: str | Path, *, filename: str | None = None
) -> Path:
    """Download an archive from a direct URL and extract it into ``destination``.

    Handles ``.zip`` and ``.tgz`` / ``.tar.gz`` / ``.tar`` (Imagenette ships as a
    ``.tgz``). Idempotent: skips the download when the archive is already present.
    Needs no credentials, so any machine with internet can provision this way.
    """
    import tarfile
    import urllib.request
    import zipfile

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    filename = filename or url.split("/")[-1].split("?")[0]
    archive = destination / filename

    if not archive.exists():
        urllib.request.urlretrieve(url, archive)  # trusted dataset URL

    lowered = filename.lower()
    if lowered.endswith((".tgz", ".tar.gz", ".tar")):
        mode = "r:gz" if lowered.endswith((".tgz", ".tar.gz")) else "r:"
        with tarfile.open(archive, mode) as tar:
            try:
                tar.extractall(destination, filter="data")  # py3.12+ safe extraction
            except TypeError:
                tar.extractall(destination)
    elif lowered.endswith(".zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive type for {filename}")
    return destination


def download_kaggle_dataset(
    slug: str, destination: str | Path, *, kind: str = "datasets", unzip: bool = True
) -> Path:
    """Download a Kaggle dataset (or competition) into ``destination``.

    Requires the ``kaggle`` package and credentials (``~/.kaggle/kaggle.json``
    or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables), so any
    machine with the token provisioned can fetch the data unattended.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Kaggle download needs the 'kaggle' package and credentials. "
            'Install with pip install kaggle and set KAGGLE_USERNAME/KAGGLE_KEY '
            "(or ~/.kaggle/kaggle.json)."
        ) from error

    api = KaggleApi()
    api.authenticate()
    if kind == "competitions":
        api.competition_download_files(slug, path=str(destination), quiet=False)
    else:
        api.dataset_download_files(slug, path=str(destination), quiet=False, unzip=unzip)
    return destination


def ensure_dataset(dataset_cfg: dict[str, Any]) -> Path:
    """Guarantee that a dataset's split manifests exist locally, provisioning if needed.

    Idempotent: returns immediately once the ``full`` train split is present.
    Otherwise it follows the config's ``provision`` block to download from Kaggle
    and/or build source manifests + splits from an ImageFolder tree.
    """
    data_root = Path(dataset_cfg["data_root"])
    splits = dataset_cfg.get("splits", {})
    full_split = Path(splits.get("full", data_root / "splits" / "full" / "train.csv"))
    if full_split.exists():
        return data_root

    provision = dataset_cfg.get("provision")
    if not provision:
        raise FileNotFoundError(
            f"Splits are missing for '{dataset_cfg.get('name')}' and no 'provision' "
            f"block is configured. Expected {full_split}. Either place the data there "
            "or add a provision block (see configs/datasets/imagenet100.yaml)."
        )

    data_root.mkdir(parents=True, exist_ok=True)
    sources = provision.get("manifest_sources", [])
    if not sources:
        raise ValueError(
            f"provision.manifest_sources is empty for '{dataset_cfg.get('name')}'; "
            "cannot build a source manifest."
        )

    def resolve_source_dir(source: dict[str, Any]) -> Path:
        path = Path(source["path"])
        return path if path.is_absolute() else data_root / path

    # Download only if the image folders aren't already on disk (e.g. a Kaggle
    # notebook mounts them read-only, so nothing needs downloading).
    if not all(resolve_source_dir(source).exists() for source in sources):
        url = provision.get("url")
        competition = provision.get("competition")
        kaggle = provision.get("kaggle")
        if url:
            download_and_extract(url, data_root, filename=provision.get("filename"))
        elif competition:
            download_kaggle_dataset(
                competition, data_root, kind="competitions", unzip=provision.get("unzip", True)
            )
        elif kaggle:
            download_kaggle_dataset(
                kaggle["slug"],
                data_root,
                kind=kaggle.get("kind", "datasets"),
                unzip=kaggle.get("unzip", True),
            )
        else:
            missing = [
                str(resolve_source_dir(s)) for s in sources if not resolve_source_dir(s).exists()
            ]
            raise FileNotFoundError(
                f"Image sources not found and no 'url'/'competition'/'kaggle' download is "
                f"configured for '{dataset_cfg.get('name')}': {missing}"
            )

    # Store image paths relative to the per-device image_root so the resulting
    # splits are portable and can be committed once and reused on any machine.
    image_root = Path(dataset_cfg.get("image_root", data_root))
    frames: list[pd.DataFrame] = []
    frames_by_role: dict[str, list[pd.DataFrame]] = {}
    for source in sources:
        directory = resolve_source_dir(source)
        absolute = bool(source.get("absolute", False))
        include_classes = None
        if source.get("class_list_file"):
            include_classes = [
                line.strip()
                for line in Path(source["class_list_file"]).read_text().splitlines()
                if line.strip()
            ]
        try:
            frame = build_image_manifest(
                directory,
                path_relative_to=None if absolute else image_root,
                split_prefix=source.get("prefix"),
                include_classes=include_classes,
                max_classes=source.get("max_classes"),
                absolute_paths=absolute,
            )
        except ValueError as error:
            raise ValueError(
                f"Cannot store paths for source {directory} relative to image_root "
                f"{image_root}. Set dataset 'image_root' to a parent of the image "
                f"folders, or set the source's 'absolute: true'. ({error})"
            ) from error
        frames.append(frame)
        if source.get("role"):
            frames_by_role.setdefault(str(source["role"]), []).append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    manifest_path = write_source_manifest(manifest, data_root / "source_manifest.csv")

    # Freeze the exact class selection next to the split so it is committable and
    # auditable (e.g. the 100 chosen wnids), one label per line, sorted.
    class_list = sorted(manifest["label"].astype(str).unique())
    (data_root / "class_list.txt").write_text("\n".join(class_list) + "\n", encoding="utf-8")

    from synthbench.datasets import create_nested_subsets

    fractions = tuple(dataset_cfg.get("real_fractions", (0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0)))
    seed = int(provision.get("seed", 42))
    if frames_by_role:
        # Role mode (e.g. Imagenette): use the dataset's own train/ and val/
        # folders directly instead of carving. val doubles as the test set unless
        # a 'test' role is given.
        _write_role_splits(
            frames_by_role, data_root, seed=seed, fractions=fractions,
            create_nested_subsets=create_nested_subsets,
        )
    else:
        # Pool-and-carve mode (e.g. ImageNet-100 from unlabelled competition val).
        _write_splits_from_manifest(
            manifest,
            data_root,
            seed=seed,
            validation_fraction=float(provision.get("validation_fraction", 0.1)),
            test_fraction=float(provision.get("test_fraction", 0.1)),
            fractions=fractions,
            create_nested_subsets=create_nested_subsets,
        )
    if not full_split.exists():
        raise RuntimeError(
            f"Provisioning ran but {full_split} is still missing; built manifest at "
            f"{manifest_path}. Check the split paths in the dataset config."
        )
    return data_root


_FRACTION_NAMES = {
    0.01: "1pct",
    0.03: "3pct",
    0.05: "5pct",
    0.1: "10pct",
    0.25: "25pct",
    0.5: "50pct",
    1.0: "full",
}


def _write_role_splits(
    frames_by_role: dict[str, list[pd.DataFrame]],
    data_root: Path,
    *,
    seed: int,
    fractions: tuple[float, ...],
    create_nested_subsets,
) -> None:
    """Assign splits from explicit source roles rather than carving them.

    ``train`` sources become the training pool (and its nested subsets), ``val``
    sources become the validation split, and ``test`` sources the test split. If
    no ``test`` role is given, validation doubles as test.
    """
    if "train" not in frames_by_role:
        raise ValueError("Role-based provisioning needs at least one source with role: train")

    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train = pd.concat(frames_by_role["train"], ignore_index=True)
    validation = pd.concat(frames_by_role["val"], ignore_index=True) if "val" in frames_by_role else None
    if "test" in frames_by_role:
        test = pd.concat(frames_by_role["test"], ignore_index=True)
    else:
        test = validation  # val doubles as test when no dedicated test source
    if validation is None:
        raise ValueError("Role-based provisioning needs a source with role: val")

    validation.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)

    subsets = create_nested_subsets(train, fractions=fractions, seed=seed, label_column="label")
    for fraction, subset in subsets.items():
        name = _FRACTION_NAMES.get(round(float(fraction), 2), f"f{int(fraction * 100)}")
        destination = split_dir / name / "train.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(destination, index=False)


def _write_splits_from_manifest(
    manifest: pd.DataFrame,
    data_root: Path,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    fractions: tuple[float, ...],
    create_nested_subsets,
) -> None:
    """Deterministic stratified train/val/test split + nested train subsets."""
    split_dir = data_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    shuffled = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_size = round(len(shuffled) * test_fraction)
    val_size = round(len(shuffled) * validation_fraction)
    test = shuffled.iloc[:test_size].copy()
    validation = shuffled.iloc[test_size : test_size + val_size].copy()
    train = shuffled.iloc[test_size + val_size :].copy()

    validation.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)

    subsets = create_nested_subsets(
        train, fractions=fractions, seed=seed, label_column="label"
    )
    for fraction, subset in subsets.items():
        name = _FRACTION_NAMES.get(round(float(fraction), 2), f"f{int(fraction * 100)}")
        destination = split_dir / name / "train.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(destination, index=False)


def kaggle_credentials_present() -> bool:
    """True when Kaggle credentials are discoverable in the environment."""
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()
