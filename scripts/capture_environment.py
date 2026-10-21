#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sclc.config import ModelProfileConfig, load_config

PACKAGES = [
    "numpy",
    "pandas",
    "pydantic",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "huggingface-hub",
    "peft",
    "pyarrow",
    "typer",
    "rich",
    "tqdm",
]
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def run(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def git_snapshot(root: Path) -> dict[str, Any]:
    return {
        "commit": run(["git", "rev-parse", "HEAD"], root),
        "status_porcelain": run(["git", "status", "--porcelain"], root),
    }


def frozen_inputs(root: Path) -> dict[str, Any]:
    paths = {
        "sample_manifest": Path("data/subsets/selected_documents.csv"),
        "query_types": Path("data/retrieval_units/query_types.csv"),
        "query_type_coding_record": Path(
            "data/retrieval_units/query_type_coding_record.csv"
        ),
        "challenge_review": Path(
            "data/subsets_cross_section_challenge/review_decisions.csv"
        ),
        "challenge_documents": Path(
            "data/subsets_cross_section_challenge/selected_documents.csv"
        ),
    }
    return {
        name: {"path": str(path), "sha256": file_sha256(root / path)}
        for name, path in paths.items()
    }


def _repo_cache_path(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def huggingface_cache_snapshot(
    cache_dir: Path,
    repo_id: str,
    configured_revision: str | None,
) -> dict[str, Any]:
    """Inspect the local Hub cache without inventing a historical revision."""
    repo_cache = _repo_cache_path(cache_dir, repo_id)
    snapshots_dir = repo_cache / "snapshots"
    available_snapshots = (
        sorted(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots_dir.is_dir()
        else []
    )

    reference = configured_revision or "main"
    resolved_commit: str | None = None
    resolution_source: str | None = None

    if configured_revision and _COMMIT_RE.fullmatch(configured_revision):
        if (snapshots_dir / configured_revision).is_dir():
            resolved_commit = configured_revision
            resolution_source = "configured_commit"
    else:
        ref_path = repo_cache / "refs" / reference
        if ref_path.is_file():
            value = ref_path.read_text(encoding="utf-8").strip()
            if _COMMIT_RE.fullmatch(value):
                resolved_commit = value
                resolution_source = f"cache_ref:{reference}"

    return {
        "cache_path": str(repo_cache),
        "cache_present": repo_cache.is_dir(),
        "reference_checked": reference,
        "resolved_commit": resolved_commit,
        "resolution_source": resolution_source,
        "available_snapshots": available_snapshots,
    }


def model_snapshot(
    model: ModelProfileConfig,
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    payload = model.model_dump()
    payload["cache"] = huggingface_cache_snapshot(
        cache_dir,
        model.model_id,
        model.revision,
    )
    adapter_source = model.adapter_source or model.model_id
    if adapter_source != model.model_id:
        payload["adapter_cache"] = huggingface_cache_snapshot(
            cache_dir,
            adapter_source,
            model.adapter_revision or model.revision,
        )
    return payload


def torch_snapshot() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"available": False}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(index))
    return {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "gpus": gpus,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility/environment.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    cache_dir = (
        root / config.paths.hf_cache_dir
        if not config.paths.hf_cache_dir.is_absolute()
        else config.paths.hf_cache_dir
    )
    payload = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": torch_snapshot(),
        "repository": git_snapshot(root),
        "frozen_inputs": frozen_inputs(root),
        "dataset": {
            "repo_id": config.dataset.repo_id,
            "subset": config.dataset.subset,
            "splits": config.dataset.splits,
            "parquet_files": config.dataset.parquet_files,
        },
        "canonical_tokenizer": {
            "model_id": config.chunking.canonical_tokenizer,
            "configured_revision": config.chunking.canonical_tokenizer_revision,
            "cache": huggingface_cache_snapshot(
                cache_dir,
                config.chunking.canonical_tokenizer,
                config.chunking.canonical_tokenizer_revision,
            ),
        },
        "models": {
            "granite": model_snapshot(config.models.granite, cache_dir=cache_dir),
            "jina": model_snapshot(config.models.jina, cache_dir=cache_dir),
        },
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(output)


if __name__ == "__main__":
    main()
