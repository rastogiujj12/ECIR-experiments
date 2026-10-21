from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from sclc.data.qasa import QASA_DATASET_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and probe the official QASA v1.1 answerable JSON.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("data/qasa/raw/testset_answerable_1554_v1.1.json"),
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def valid_json(path: Path) -> bool:
        if not path.exists() or path.stat().st_size < 1000:
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False

        def looks_like_record(value: object) -> bool:
            if not isinstance(value, dict):
                return False
            return bool(value.get("question") or value.get("input")) and bool(
                value.get("evidential_info") is not None
                or value.get("evidence") is not None
                or value.get("ctxs") is not None
                or value.get("contexts") is not None
            )

        if isinstance(data, list):
            return any(looks_like_record(item) for item in data)
        if isinstance(data, dict):
            for key in ("data", "examples"):
                value = data.get(key)
                if isinstance(value, list) and any(looks_like_record(item) for item in value):
                    return True
            return any(looks_like_record(value) for value in data.values())
        return False

    if not valid_json(args.output):
        response = requests.get(
            QASA_DATASET_URL,
            timeout=args.timeout,
            headers={"User-Agent": "sclc-retrieval/1.0 (QASA preparation)"},
        )
        response.raise_for_status()
        payload = response.content
        args.output.write_bytes(payload)
        if not valid_json(args.output):
            try:
                args.output.unlink()
            except OSError:
                pass
            raise RuntimeError("Downloaded QASA file is not a valid non-trivial JSON release.")

    print(f"QASA file ready: {args.output}")
    print("Next: run `python scripts/probe_qasa.py <path>` to inspect the released JSON schema.")


if __name__ == "__main__":
    main()
