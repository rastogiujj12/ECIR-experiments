from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from sclc.data.schema import DocumentRecord


def write_documents_jsonl(documents: Iterable[DocumentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(document.model_dump_json())
            handle.write("\n")


def read_documents_jsonl(path: Path) -> Iterator[DocumentRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield DocumentRecord.model_validate(json.loads(line))


def document_summary_frame(documents: Iterable[DocumentRecord]) -> pd.DataFrame:
    rows = []
    for document in documents:
        rows.append(
            {
                "document_id": document.document_id,
                "split": document.split,
                "title": document.title,
                "character_count": len(document.text),
                "section_count": len(document.sections),
                "paragraph_count": len(document.paragraphs),
                "question_count": len(document.queries),
                "usable_question_count": document.usable_question_count,
            }
        )
    return pd.DataFrame(rows)


def evidence_alignment_frame(documents: Iterable[DocumentRecord]) -> pd.DataFrame:
    rows = []
    for document in documents:
        for query in document.queries:
            for answer_index, answer in enumerate(query.answers):
                for evidence_index, evidence in enumerate(answer.evidence):
                    rows.append(
                        {
                            "document_id": document.document_id,
                            "query_id": query.query_id,
                            "answer_index": answer_index,
                            "annotation_id": answer.annotation_id,
                            "evidence_index": evidence_index,
                            "is_float": evidence.is_float,
                            "alignment_status": evidence.alignment_status,
                            "matched_paragraph_count": len(evidence.matched_paragraph_ids),
                            "evidence_text": evidence.text,
                        }
                    )
    return pd.DataFrame(rows)
