from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from datasets import Dataset, load_dataset

from sclc.config import AppConfig


def load_qasper_splits(config: AppConfig) -> Iterator[tuple[str, Dataset]]:
    """Load QASPER directly from its published Parquet exports.

    Current Hugging Face Datasets releases do not execute dataset scripts
    such as qasper.py. The configured Parquet URLs bypass that script
    while preserving QASPER's records and original train/validation/test splits.
    """
    for split in config.dataset.splits:
        parquet_url = config.dataset.parquet_files.get(split)
        if parquet_url is None:
            raise ValueError(
                f"No Parquet file is configured for split {split!r}. "
                "Add it under dataset.parquet_files in the YAML configuration."
            )

        dataset = load_dataset(
            "parquet",
            data_files={split: parquet_url},
            split=split,
            cache_dir=str(config.paths.hf_cache_dir),
        )
        yield split, dataset


def iter_raw_records(config: AppConfig) -> Iterable[tuple[str, dict[str, Any]]]:
    for split, dataset in load_qasper_splits(config):
        for row in dataset:
            yield split, dict(row)
