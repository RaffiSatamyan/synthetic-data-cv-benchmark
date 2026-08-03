import pandas as pd

from synthbench.generation import GenerationRequest, PlaceholderGenerator, generate_dataset


def test_placeholder_generation_is_resumable(tmp_path):
    requests = [
        GenerationRequest(sample_id=f"sample_{index}", seed=index, label=index % 2)
        for index in range(3)
    ]
    manifest = generate_dataset(
        PlaceholderGenerator(), requests, tmp_path / "generation", {"test": True}
    )
    generate_dataset(
        PlaceholderGenerator(), requests, tmp_path / "generation", {"test": True}, resume=True
    )

    frame = pd.read_csv(manifest)
    assert len(frame) == 3
    assert frame.sample_id.is_unique
    assert all((tmp_path / "generation" / path).exists() for path in frame.image_path)
