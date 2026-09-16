import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from sclc.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_qasper_parquet_urls_are_configured() -> None:
    config = load_config(REPO_ROOT / "configs/base.yaml")

    assert set(config.dataset.parquet_files) == {"train", "validation", "test"}
    assert all(
        url.startswith("https://huggingface.co/datasets/allenai/qasper/resolve/")
        for url in config.dataset.parquet_files.values()
    )


def test_qasper_loader_uses_configured_parquet_splits(monkeypatch: Any) -> None:
    config = load_config(REPO_ROOT / "configs/base.yaml")
    calls: list[dict[str, Any]] = []

    def fake_load_dataset(
        dataset_format: str,
        *,
        data_files: dict[str, str],
        split: str,
        cache_dir: str,
    ) -> list[dict[str, str]]:
        calls.append(
            {
                "dataset_format": dataset_format,
                "data_files": data_files,
                "split": split,
                "cache_dir": cache_dir,
            }
        )
        return [{"id": split}]

    fake_datasets = ModuleType("datasets")
    fake_datasets.Dataset = object
    fake_datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    sys.modules.pop("sclc.data.qasper", None)
    qasper = importlib.import_module("sclc.data.qasper")

    loaded = list(qasper.load_qasper_splits(config))

    assert [split for split, _ in loaded] == ["train", "validation", "test"]
    assert [call["split"] for call in calls] == ["train", "validation", "test"]
    assert all(call["dataset_format"] == "parquet" for call in calls)
    assert all(call["cache_dir"] == str(config.paths.hf_cache_dir) for call in calls)
    for call in calls:
        split = str(call["split"])
        assert call["data_files"] == {split: config.dataset.parquet_files[split]}
