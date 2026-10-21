from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def shape(value: Any, depth: int = 0, max_depth: int = 2) -> Any:
    if depth >= max_depth:
        if isinstance(value, list):
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            return f"dict[{len(value)}]"
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): shape(item, depth + 1, max_depth) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [shape(item, depth + 1, max_depth) for item in value[:2]]
    return value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__


def preview(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else text[:limit] + "\n...<truncated>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect QASA JSON schema and show a small sample of the evidence annotation."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    with args.path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    print("root:")
    print(shape(raw))

    record = None
    if isinstance(raw, list) and raw:
        record = raw[0]
    elif isinstance(raw, dict):
        if isinstance(raw.get("data"), list) and raw["data"]:
            record = raw["data"][0]
        elif isinstance(raw.get("examples"), list) and raw["examples"]:
            record = raw["examples"][0]
        else:
            values = list(raw.values())
            record = values[0] if values and isinstance(values[0], dict) else None

    if not isinstance(record, dict):
        return

    print("\nfirst record keys:")
    print(sorted(record.keys()))

    print("\nfirst record selected fields:")
    selected = {
        key: record.get(key)
        for key in (
            "paper_id",
            "title",
            "question_id",
            "question",
            "question_section",
            "question_type",
            "evidential_info",
            "composition",
            "arxiv_id",
            "s2orc_url",
            "arxiv_url",
        )
        if key in record
    }
    print(preview(selected))

    evidential = record.get("evidential_info")
    print("\nevidential_info shape:")
    print(shape(evidential, max_depth=4))
    print("\nevidential_info preview:")
    print(preview(evidential))


if __name__ == "__main__":
    main()
