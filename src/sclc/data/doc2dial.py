from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from sclc.config import AppConfig
from sclc.data.reconstruct import normalize_text
from sclc.data.schema import (
    AnswerAnnotation,
    CharacterSpan,
    DocumentRecord,
    EvidenceItem,
    ParagraphRecord,
    QueryRecord,
    SectionRecord,
)

DATA_URL = "https://doc2dial.github.io/file/doc2dial_v1.0.1.zip"


def download_doc2dial(cache_dir: Path) -> Path:
    """Download and extract the official Doc2Dial v1.0 archive."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "doc2dial_v1.0.1.zip"
    root = cache_dir / "doc2dial" / "v1.0.1"
    if not root.exists():
        if not archive.exists():
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cache_dir)
    if not (root / "doc2dial_doc.json").exists():
        raise FileNotFoundError(
            f"Expected {root / 'doc2dial_doc.json'} after extracting Doc2Dial v1.0.1"
        )
    return root


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)



def _normalise_parent_titles(value: Any) -> list[str]:
    """Normalize Doc2Dial's hierarchy annotation without altering document text.

    The published data schema exposes ``parent_titles`` as a string, while the
    underlying JSON uses a list-like value.  Accept both representations so the
    adapter consumes the source annotation as-is.
    """
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            import ast
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = [stripped]
        value = parsed
    if isinstance(value, (tuple, list)):
        titles: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = normalize_text(str(item.get("text") or item.get("title") or ""))
            else:
                text = normalize_text(str(item))
            if text:
                titles.append(text)
        return titles
    text = normalize_text(str(value))
    return [text] if text else []


def _raw_spans(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_value = raw.get("spans") or {}
    if isinstance(raw_value, dict):
        result: list[dict[str, Any]] = []
        for span_id, item in raw_value.items():
            if isinstance(item, dict):
                span = dict(item)
                span.setdefault("id_sp", span_id)
                result.append(span)
        return result
    return [dict(item) for item in raw_value if isinstance(item, dict)]


def _hierarchy_for_span(item: dict[str, Any]) -> list[str]:
    parents = _normalise_parent_titles(item.get("parent_titles"))
    title_text = normalize_text(str(item.get("title") or ""))
    hierarchy = [*parents, title_text] if title_text else parents
    return [value for value in hierarchy if value]


def _normalize_disjoint_top_sections(
    sections: list[dict[str, Any]],
    document_length: int,
) -> list[dict[str, Any]]:
    """Turn source hierarchy starts into a disjoint top-level partition.

    This does not clean, rewrite, or infer content. It only converts the
    publisher-provided hierarchy into the non-overlapping interval form required
    by the existing chunker.
    """
    cleaned: list[dict[str, Any]] = []
    for sec in sections:
        start = max(0, min(int(sec["start"]), document_length))
        end = max(start, min(int(sec["end"]), document_length))
        title = str(sec.get("title", "Document")).strip() or "Document"
        if start >= end:
            continue
        item = dict(sec)
        item["start"] = start
        item["end"] = end
        item["title"] = title
        cleaned.append(item)

    cleaned.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["title"]))
    by_start: dict[int, dict[str, Any]] = {}
    for item in cleaned:
        by_start.setdefault(item["start"], item)
    starts = sorted(by_start)
    result: list[dict[str, Any]] = []
    for i, start in enumerate(starts):
        item = dict(by_start[start])
        next_start = starts[i + 1] if i + 1 < len(starts) else document_length
        item["start"] = start
        item["end"] = next_start
        if item["end"] > item["start"]:
            result.append(item)
    return result


def _build_document(
    *,
    domain: str,
    doc_id: str,
    raw: dict[str, Any],
    query_records: list[QueryRecord],
) -> DocumentRecord:
    """Convert a Doc2Dial source document without content preprocessing."""
    text = str(raw.get("doc_text") or "")
    title = normalize_text(str(raw.get("title") or ""))
    raw_spans = _raw_spans(raw)

    valid_spans: list[dict[str, Any]] = []
    for item in raw_spans:
        start = int(item.get("start_sp", 0))
        end = int(item.get("end_sp", 0))
        if 0 <= start < end <= len(text):
            valid_spans.append(item)
    valid_spans.sort(key=lambda item: (int(item["start_sp"]), int(item["end_sp"]), str(item.get("id_sp", ""))))

    # Use the dataset's explicit section annotations: each unique ``id_sec`` is
    # treated as one structural scope.  ``start_sec``/``end_sec`` and the section
    # title are supplied by Doc2Dial itself.  No HTML parsing or text-derived
    # section boundary inference is performed here.
    source_sections: dict[str, dict[str, Any]] = {}
    for item in valid_spans:
        section_key = str(item.get("id_sec") or item.get("id_sp") or "")
        if not section_key:
            continue
        start = int(item.get("start_sec", item["start_sp"]))
        end = int(item.get("end_sec", item["end_sp"]))
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        hierarchy = _hierarchy_for_span(item)
        heading = str(item.get("title") or "")
        existing = source_sections.get(section_key)
        if existing is None:
            source_sections[section_key] = {
                "title": normalize_text(heading) or (hierarchy[-1] if hierarchy else title or "Document"),
                "start": start,
                "end": end,
            }
        else:
            existing["start"] = min(existing["start"], start)
            existing["end"] = max(existing["end"], end)
            if not existing["title"] and heading:
                existing["title"] = normalize_text(heading)

    runs = sorted(source_sections.values(), key=lambda item: (item["start"], item["end"], item["title"]))
    runs = _normalize_disjoint_top_sections(runs, len(text))
    if not runs and text:
        runs = [{"title": title or "Document", "start": 0, "end": len(text)}]

    section_ids = [f"{domain}:{doc_id}:section:{i:04d}" for i in range(len(runs))]
    sections: list[SectionRecord] = []
    for index, run in enumerate(runs):
        sections.append(
            SectionRecord(
                section_id=section_ids[index],
                source_section_index=index,
                heading=str(run["title"]),
                top_level_heading=str(run["title"]),
                hierarchy=[str(run["title"])],
                span=CharacterSpan(start=int(run["start"]), end=int(run["end"])),
                paragraph_ids=[],
            )
        )

    def section_for_span(start: int, end: int) -> int | None:
        for index, run in enumerate(runs):
            if run["start"] <= start < run["end"]:
                return index
        for index, run in enumerate(runs):
            overlap = max(0, min(end, run["end"]) - max(start, run["start"]))
            if overlap > 0:
                return index
        return None

    paragraphs: list[ParagraphRecord] = []
    for index, item in enumerate(valid_spans):
        start = int(item["start_sp"])
        end = int(item["end_sp"])
        section_index = section_for_span(start, end)
        if section_index is None:
            continue
        paragraph = ParagraphRecord(
            paragraph_id=f"{domain}:{doc_id}:paragraph:{index:05d}",
            section_id=section_ids[section_index],
            source_section_index=section_index,
            source_paragraph_index=index,
            text=text[start:end],
            span=CharacterSpan(start=start, end=end),
        )
        paragraphs.append(paragraph)
        sections[section_index].paragraph_ids.append(paragraph.paragraph_id)

    return DocumentRecord(
        document_id=f"{domain}:{doc_id}",
        split="validation",
        title=title,
        abstract="",
        text=text,
        sections=sections,
        paragraphs=paragraphs,
        queries=query_records,
    )


def _make_query_records(
    *,
    domain: str,
    doc_id: str,
    dialogues: list[dict[str, Any]],
    spans: dict[str, dict[str, Any]],
) -> list[QueryRecord]:
    queries: list[QueryRecord] = []
    for dialogue in dialogues:
        turns = dialogue.get("turns") or []
        for index, turn in enumerate(turns):
            if turn.get("role") != "user" or index + 1 >= len(turns):
                continue
            next_turn = turns[index + 1]
            if next_turn.get("role") != "agent":
                continue

            references = [
                ref for ref in (next_turn.get("references") or [])
                if ref.get("label") in {"precondition", "solution"}
            ]
            if not references:
                continue

            evidence: list[EvidenceItem] = []
            for ref in references:
                span = spans.get(str(ref.get("sp_id")))
                if span is None:
                    continue
                text = str(span.get("text_sp") or "")
                if not text.strip():
                    continue
                evidence.append(
                    EvidenceItem(
                        text=text,
                        is_float=False,
                        alignment_status="pending",
                    )
                )
            if not evidence:
                continue

            query_id = f"{domain}:{dialogue['dial_id']}:{turn['turn_id']}"
            queries.append(
                QueryRecord(
                    query_id=query_id,
                    question=normalize_text(str(turn.get("utterance") or "")),
                    answers=[AnswerAnnotation(annotation_id=query_id, evidence=evidence)],
                )
            )
    return queries


def iter_doc2dial_documents(
    config: AppConfig,
    *,
    split: str = "validation",
) -> Iterator[DocumentRecord]:
    """Yield Doc2Dial validation documents grounded by official dialogue references."""
    root = download_doc2dial(config.paths.hf_cache_dir / "doc2dial")
    documents = _load_json(root / "doc2dial_doc.json")["doc_data"]
    dial_filename = {
        "train": "doc2dial_dial_train.json",
        "validation": "doc2dial_dial_validation.json",
    }.get(split)
    if dial_filename is None:
        raise ValueError("Doc2Dial supports train and validation dialogue splits in v1.0.1")
    dial_data = _load_json(root / dial_filename)["dial_data"]

    queries_by_doc: dict[tuple[str, str], list[QueryRecord]] = defaultdict(list)
    for domain, doc_dialogues in dial_data.items():
        for doc_id, dialogues in doc_dialogues.items():
            raw_doc = documents[domain][doc_id]
            raw_spans = _raw_spans(raw_doc)
            span_lookup = {str(item["id_sp"]): item for item in raw_spans if "id_sp" in item}
            queries_by_doc[(domain, doc_id)].extend(
                _make_query_records(
                    domain=domain,
                    doc_id=doc_id,
                    dialogues=dialogues,
                    spans=span_lookup,
                )
            )

    for domain, domain_documents in documents.items():
        for doc_id, raw_doc in domain_documents.items():
            queries = queries_by_doc.get((domain, doc_id), [])
            if not queries:
                continue
            document = _build_document(
                domain=domain,
                doc_id=doc_id,
                raw=raw_doc,
                query_records=queries,
            )

            lookup: dict[str, list[str]] = defaultdict(list)
            for paragraph in document.paragraphs:
                lookup[normalize_text(paragraph.text)].append(paragraph.paragraph_id)
            for query in document.queries:
                for answer in query.answers:
                    for evidence in answer.evidence:
                        ids = lookup.get(normalize_text(evidence.text), [])
                        evidence.matched_paragraph_ids = ids
                        evidence.alignment_status = (
                            "matched_unique" if len(ids) == 1
                            else "matched_ambiguous" if len(ids) > 1
                            else "unmatched"
                        )
            yield document
