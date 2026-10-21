#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

USABLE_STATUSES = {
    "matched_unique",
    "matched_fuzzy_unique",
    "matched_fuzzy_scattered",
    "matched_span",
    "matched_fuzzy_span",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.values()) if isinstance(data, dict) else list(data)


def canonical_raw_key(item: dict[str, Any]) -> tuple[str, str]:
    doc_id = item.get("arxiv_id") or item.get("paper_id")
    return str(doc_id), str(item.get("question_id"))


def canonical_csv_key(row: dict[str, str]) -> tuple[str, str]:
    document_id = str(row.get("document_id"))
    query_id = str(row.get("query_id") or row.get("question_id"))

    # Retrieval query IDs are globally unique after the QASA adapter fix,
    # using the form ``<document_id>:<local_question_id>``.  The raw QASA
    # release, however, stores only the local numeric question_id.
    # Normalize the alignment CSV back to the raw-QASA key for manifesting.
    prefix = f"{document_id}:"
    if query_id.startswith(prefix):
        query_id = query_id[len(prefix):]

    return document_id, query_id


def parse_ids(row: dict[str, str], *names: str) -> list[str]:
    for name in names:
        value = row.get(name)
        if value:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            return [x.strip() for x in value.split(";") if x.strip()]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the validated QASA retrieval evaluation set.")
    parser.add_argument("--raw", default="data/qasa/raw/testset_answerable_1554_v1.1.json")
    parser.add_argument("--alignment", default="data/qasa/processed/evidence_alignment.csv")
    parser.add_argument("--out-dir", default="data/qasa/processed")
    parser.add_argument("--expected-raw", type=int, default=1554)
    parser.add_argument("--expected-eligible", type=int, default=1408)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    alignment_path = Path(args.alignment)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_json(raw_path)
    raw_by_key = {canonical_raw_key(x): x for x in raw_records}

    with alignment_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {"document_id", "alignment_status"}
    missing_cols = required - set(rows[0]) if rows else required
    if missing_cols:
        raise RuntimeError(f"Alignment CSV is missing required columns: {sorted(missing_cols)}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[canonical_csv_key(row)].append(row)

    represented = set(grouped)
    raw_keys = set(raw_by_key)
    missing_from_csv = sorted(raw_keys - represented)

    eligible_keys: list[tuple[str, str]] = []
    excluded_keys: list[tuple[str, str]] = []

    for key, evidence_rows in grouped.items():
        statuses = [r.get("alignment_status", "") for r in evidence_rows]
        if statuses and all(status in USABLE_STATUSES for status in statuses):
            eligible_keys.append(key)
        else:
            excluded_keys.append(key)

    eligible_keys.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))
    excluded_keys.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))

    if args.expected_raw is not None and len(raw_records) != args.expected_raw:
        raise RuntimeError(f"Expected {args.expected_raw} raw records, found {len(raw_records)}")

    if args.expected_eligible is not None and len(eligible_keys) != args.expected_eligible:
        raise RuntimeError(
            f"Expected {args.expected_eligible} eligible questions, found {len(eligible_keys)}. "
            "Refusing to freeze a different evaluation set."
        )

    eligible_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []

    for key in eligible_keys:
        raw = raw_by_key.get(key)
        if raw is None:
            raise RuntimeError(f"Eligible CSV question has no raw QASA record: {key}")
        evidence = []
        raw_evidence = raw.get("evidential_info") or []
        for row in grouped[key]:
            idx_raw = row.get("evidence_index")
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                idx = None
            source_ev = raw_evidence[idx] if idx is not None and 0 <= idx < len(raw_evidence) else {}
            evidence.append({
                "evidence_index": idx_raw,
                "alignment_status": row.get("alignment_status"),
                "matched_paragraph_ids": parse_ids(row, "matched_paragraph_ids", "matched_ids"),
                "context": source_ev.get("context", ""),
                "rationale": source_ev.get("rationale", ""),
            })
        eligible_records.append({
            "paper_id": raw.get("paper_id"),
            "document_id": key[0],
            "arxiv_id": raw.get("arxiv_id"),
            "s2orc_id": raw.get("s2orc_id"),
            "title": raw.get("title"),
            "question_id": raw.get("question_id"),
            "question": raw.get("question"),
            "question_section": raw.get("question_section"),
            "question_type": raw.get("question_type"),
            "composition": raw.get("composition"),
            "evidence": evidence,
        })

    for key in excluded_keys:
        raw = raw_by_key.get(key)
        evidence_rows = grouped[key]
        excluded_records.append({
            "paper_id": raw.get("paper_id") if raw else None,
            "document_id": key[0],
            "question_id": raw.get("question_id") if raw else key[1],
            "title": raw.get("title") if raw else None,
            "question": raw.get("question") if raw else None,
            "statuses": [r.get("alignment_status") for r in evidence_rows],
        })

    missing_records = []
    for key in missing_from_csv:
        raw = raw_by_key[key]
        missing_records.append({
            "paper_id": raw.get("paper_id"),
            "document_id": key[0],
            "question_id": raw.get("question_id"),
            "arxiv_id": raw.get("arxiv_id"),
            "title": raw.get("title"),
            "question": raw.get("question"),
            "reason": "No alignment CSV rows; inspect raw QASA evidence."
        })

    eligible_path = out_dir / "qasa_eligible_questions.jsonl"
    with eligible_path.open("w", encoding="utf-8") as f:
        for record in eligible_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    excluded_path = out_dir / "qasa_excluded_questions.jsonl"
    with excluded_path.open("w", encoding="utf-8") as f:
        for record in excluded_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    missing_path = out_dir / "qasa_missing_from_alignment.jsonl"
    with missing_path.open("w", encoding="utf-8") as f:
        for record in missing_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "dataset": "QASA",
        "release": "v1.1 answerable",
        "raw_question_count": len(raw_records),
        "represented_question_count": len(represented),
        "missing_from_alignment_count": len(missing_from_csv),
        "excluded_question_count": len(excluded_records),
        "eligible_question_count": len(eligible_records),
        "eligible_rate_of_raw": len(eligible_records) / len(raw_records),
        "eligible_rate_of_represented": len(eligible_records) / len(represented),
        "usable_alignment_statuses": sorted(USABLE_STATUSES),
        "gold_evidence_policy": "All evidence rows for a question must have an accepted paragraph alignment status; questions with incomplete or ambiguous evidence are excluded.",
        "raw_sha256": sha256_file(raw_path),
        "alignment_csv_sha256": sha256_file(alignment_path),
        "eligible_questions_sha256": sha256_file(eligible_path),
    }

    manifest_path = out_dir / "qasa_eligibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== QASA frozen evaluation set ===")
    print(f"Raw questions:              {len(raw_records)}")
    print(f"Represented in alignment:  {len(represented)}")
    print(f"Missing from alignment:    {len(missing_from_csv)}")
    print(f"Excluded:                   {len(excluded_records)}")
    print(f"Eligible / frozen:          {len(eligible_records)}")
    print(f"Eligible rate (raw):        {len(eligible_records) / len(raw_records):.2%}")
    print(f"Eligible rate (represented):{len(eligible_records) / len(represented):.2%}")
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Eligible questions: {eligible_path}")
    print(f"Excluded questions: {excluded_path}")
    print(f"Missing from alignment: {missing_path}")


if __name__ == "__main__":
    main()
