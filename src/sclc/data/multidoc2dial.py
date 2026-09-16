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

DATA_URL = "https://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip"


def download_multidoc2dial(cache_dir: Path) -> Path:
    """Download and extract the official MultiDoc2Dial v1.0 archive."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "multidoc2dial.zip"
    root = cache_dir / "multidoc2dial"
    if not root.exists():
        if not archive.exists():
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(cache_dir)
    if not (root / "multidoc2dial_doc.json").exists():
        raise FileNotFoundError(
            f"Expected {root / 'multidoc2dial_doc.json'} after extracting MultiDoc2Dial"
        )
    return root


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_parent_titles(value: Any) -> list[str]:
    titles: list[str] = []
    for item in value or []:
        if isinstance(item, dict):
            text = normalize_text(str(item.get("text") or ""))
        else:
            text = normalize_text(str(item))
        if text:
            titles.append(text)
    return titles


def _build_document(
    *,
    domain: str,
    doc_id: str,
    raw: dict[str, Any],
    split: str,
    query_records: list[QueryRecord],
) -> DocumentRecord:
    """Convert one MultiDoc2Dial document to the SCLC document schema.

    MultiDoc2Dial's ``spans`` contain *nested* subsection ranges.  A raw span's
    ``start_sec``/``end_sec`` therefore cannot be used directly as a
    ``SectionRecord``: doing so creates overlapping sections and violates the
    non-overlapping top-level-section invariant used by the SCLC chunker.

    We instead derive contiguous top-level sections from the hierarchy encoded
    in ``parent_titles``/``title`` and retain the original span offsets for
    evidence alignment.
    """
    text = str(raw.get("doc_text") or "")
    title = normalize_text(str(raw.get("title") or ""))

    raw_spans_value = raw.get("spans") or {}
    if isinstance(raw_spans_value, dict):
        raw_spans: list[dict[str, Any]] = []
        for span_id, item in raw_spans_value.items():
            if isinstance(item, dict):
                span = dict(item)
                span.setdefault("id_sp", span_id)
                raw_spans.append(span)
    else:
        raw_spans = [item for item in raw_spans_value if isinstance(item, dict)]

    # Keep valid source spans in document order.  These are used as evidence
    # units; their offsets are supplied by the dataset and are not re-parsed.
    valid_spans: list[dict[str, Any]] = []
    for item in raw_spans:
        start = int(item.get("start_sp", 0))
        end = int(item.get("end_sp", 0))
        if 0 <= start < end <= len(text):
            valid_spans.append(item)
    valid_spans.sort(key=lambda item: (int(item["start_sp"]), int(item["end_sp"])))

    def hierarchy_for(item: dict[str, Any]) -> list[str]:
        parents = _normalise_parent_titles(item.get("parent_titles"))
        title_text = normalize_text(str(item.get("title") or ""))
        hierarchy = [*parents, title_text] if title_text else parents
        return [value for value in hierarchy if value]

    # Derive non-overlapping contiguous top-level runs.  A top-level section is
    # identified by the first hierarchy component, while its character span is
    # the union of all nested source spans belonging to that run.
    runs: list[dict[str, Any]] = []
    for item in valid_spans:
        hierarchy = hierarchy_for(item)
        top_level = hierarchy[0] if hierarchy else title or "Document"
        start_sec = int(item.get("start_sec", item["start_sp"]))
        end_sec = int(item.get("end_sec", item["end_sp"]))
        start_sec = max(0, min(start_sec, len(text)))
        end_sec = max(start_sec, min(end_sec, len(text)))
        start_sp = int(item["start_sp"])

        # Find the most recent run with the same top-level heading only when
        # there is no intervening top-level heading.  This preserves separate
        # repeated headings while merging nested subsection records.
        if runs and runs[-1]["top_level"].casefold() == top_level.casefold():
            runs[-1]["start"] = min(runs[-1]["start"], start_sec)
            runs[-1]["end"] = max(runs[-1]["end"], end_sec)
            runs[-1]["last_start_sp"] = max(runs[-1]["last_start_sp"], start_sp)
        else:
            runs.append(
                {
                    "top_level": top_level,
                    "start": start_sec,
                    "end": end_sec,
                    "last_start_sp": start_sp,
                }
            )

    # MultiDoc2Dial source section ranges can overlap or nest.  Normalize the
    # derived top-level runs into a disjoint document-ordered partition before
    # passing them to the existing chunking layer.  Evidence offsets themselves
    # are never modified.
    runs = _normalize_disjoint_top_sections(runs, len(text))

    def section_for_span(item: dict[str, Any]) -> int | None:
        start = int(item["start_sp"])
        end = int(item["end_sp"])
        candidates = [
            (index, run)
            for index, run in enumerate(runs)
            if run["start"] <= start < run["end"]
            or run["start"] < end <= run["end"]
            or (start <= run["start"] and run["end"] <= end)
        ]
        if not candidates:
            return None
        # Prefer the section containing the span start, then maximum overlap.
        containing = [item for item in candidates if item[1]["start"] <= start < item[1]["end"]]
        if containing:
            return containing[0][0]
        return max(
            candidates,
            key=lambda pair: max(
                0,
                min(end, pair[1]["end"]) - max(start, pair[1]["start"]),
            ),
        )[0]

    section_ids = [f"{domain}:{doc_id}:section:{i:04d}" for i in range(len(runs))]
    sections: list[SectionRecord] = []
    for index, run in enumerate(runs):
        section_ids[index] = f"{domain}:{doc_id}:section:{index:04d}"
        sections.append(
            SectionRecord(
                section_id=section_ids[index],
                source_section_index=index,
                heading=str(run["top_level"]),
                top_level_heading=str(run["top_level"]),
                hierarchy=[str(run["top_level"])],
                span=CharacterSpan(start=int(run["start"]), end=int(run["end"])),
                paragraph_ids=[],
            )
        )

    paragraphs: list[ParagraphRecord] = []
    for index, item in enumerate(valid_spans):
        start = int(item["start_sp"])
        end = int(item["end_sp"])
        section_index = section_for_span(item)
        if section_index is None or section_index >= len(sections):
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
        split=split,
        title=title,
        abstract="",
        text=text,
        sections=sections,
        paragraphs=paragraphs,
        queries=query_records,
    )

def _query_records_for_dialogues(
    *,
    domain: str,
    doc_id: str,
    dialogues: list[dict[str, Any]],
    span_lookup: dict[str, dict[str, Any]],
) -> dict[str, list[QueryRecord]]:
    by_doc: dict[str, list[QueryRecord]] = defaultdict(list)
    for dialogue in dialogues:
        turns = dialogue.get("turns") or []
        for index, turn in enumerate(turns):
            if turn.get("role") != "user" or index + 1 >= len(turns):
                continue
            next_turn = turns[index + 1]
            if next_turn.get("role") != "agent":
                continue
            references = [
                ref
                for ref in (next_turn.get("references") or [])
                if ref.get("doc_id") == doc_id and ref.get("label") in {
                    "precondition",
                    "solution",
                }
            ]
            if not references:
                continue

            evidence: list[EvidenceItem] = []
            for ref in references:
                span = span_lookup.get(str(ref.get("id_sp")))
                if span is None:
                    continue
                start = int(span["start_sp"])
                end = int(span["end_sp"])
                evidence.append(
                    EvidenceItem(
                        text=str(span["text_sp"]),
                        is_float=False,
                        matched_paragraph_ids=[],
                        alignment_status="pending",
                    )
                )

            if not evidence:
                continue
            query_id = f"{domain}:{dialogue['dial_id']}:{turn['turn_id']}"
            by_doc[doc_id].append(
                QueryRecord(
                    query_id=query_id,
                    question=normalize_text(str(turn.get("utterance") or "")),
                    answers=[
                        AnswerAnnotation(
                            annotation_id=query_id,
                            evidence=evidence,
                        )
                    ],
                )
            )
    return by_doc


def _normalize_disjoint_top_sections(
    sections: list[dict],
    document_length: int,
) -> list[dict]:
    """Return disjoint, document-ordered top-level section intervals.

    MultiDoc2Dial contains section/span annotations whose ranges can overlap
    because nested structural annotations are represented independently.
    The SCLC chunking layer requires top-level section boundaries to form a
    non-overlapping partition.  We therefore derive ownership by document
    order: each section owns the interval from its start to the next distinct
    top-level section start (or the document end).  Duplicate/nested source
    spans are collapsed before this step.
    """
    cleaned = []
    seen = set()

    for sec in sections:
        start = max(0, min(int(sec["start"]), document_length))
        end = max(start, min(int(sec["end"]), document_length))
        title = str(sec.get("title", sec.get("top_level", ""))).strip()
        key = (start, end, title)
        if key in seen or start >= end:
            continue
        seen.add(key)
        item = dict(sec)
        item["start"] = start
        item["end"] = end
        cleaned.append(item)

    # Prefer the earliest occurrence for a structural title at the same start.
    # Sort by start, then prefer the widest annotation at a duplicate start.
    cleaned.sort(key=lambda x: (x["start"], -(x["end"] - x["start"]), x.get("title", "")))

    # Collapse identical-start entries.  The widest interval is the parent/top-level
    # representation; narrower nested entries must not become independent boundaries.
    starts = []
    by_start = {}
    for sec in cleaned:
        s = sec["start"]
        if s not in by_start:
            by_start[s] = sec
            starts.append(s)

    starts.sort()
    result = []
    for i, start in enumerate(starts):
        sec = dict(by_start[start])
        next_start = starts[i + 1] if i + 1 < len(starts) else document_length
        end = min(sec["end"], next_start) if sec["end"] > start else next_start

        # If the source span ended before the next top-level start, retain it.
        # If source spans overlap, ownership stops at the next start, making the
        # resulting intervals disjoint.
        if end <= start:
            continue
        sec["end"] = end
        result.append(sec)

    # Fill any gap between adjacent structural intervals with the preceding section.
    # This preserves the existing chunker's assumption that section membership covers
    # the prepared document rather than leaving accidental overlap/holes.
    for i in range(len(result) - 1):
        result[i]["end"] = result[i + 1]["start"]

    if result:
        result[-1]["end"] = document_length

    # Final defensive pass: enforce strict disjointness and ordering.
    final = []
    cursor = 0
    for sec in result:
        start = max(cursor, sec["start"])
        end = max(start, sec["end"])
        if end <= start:
            continue
        sec = dict(sec)
        sec.setdefault("top_level", title)
        sec["start"] = start
        sec["end"] = min(end, document_length)
        if sec["end"] > sec["start"]:
            final.append(sec)
            cursor = sec["end"]

    return final

def iter_multidoc2dial_documents(
    config: AppConfig,
    *,
    split: str = "validation",
) -> Iterator[DocumentRecord]:
    """Yield MultiDoc2Dial documents in the existing SCLC DocumentRecord format.

    Validation is used by default because the official v1.0 archive's top-level
    test file is a dummy file; the underlying domain files contain train/dev/test
    splits, but they require a domain-specific assembly step.
    """
    root = download_multidoc2dial(config.paths.hf_cache_dir / "multidoc2dial")
    documents = _load_json(root / "multidoc2dial_doc.json")["doc_data"]

    dial_filename = {
        "train": "multidoc2dial_dial_train.json",
        "validation": "multidoc2dial_dial_validation.json",
    }.get(split)
    if dial_filename is None:
        raise ValueError(
            "MultiDoc2Dial v1.0 does not provide a usable top-level test file. "
            "Use train or validation, or add an explicit domain-test assembly."
        )
    dial_data = _load_json(root / dial_filename)["dial_data"]

    queries_by_doc: dict[tuple[str, str], list[QueryRecord]] = defaultdict(list)
    for domain, dialogues in dial_data.items():
        for dialogue in dialogues:
            turns = dialogue.get("turns") or []
            for index, turn in enumerate(turns):
                if turn.get("role") != "user" or index + 1 >= len(turns):
                    continue
                next_turn = turns[index + 1]
                if next_turn.get("role") != "agent":
                    continue
                refs = [
                    ref for ref in (next_turn.get("references") or [])
                    if ref.get("label") in {"precondition", "solution"}
                ]
                if not refs:
                    continue
                doc_id = str(refs[0]["doc_id"])
                raw_doc = documents[domain][doc_id]
                raw_spans = raw_doc.get("spans") or {}
                if isinstance(raw_spans, dict):
                    span_lookup = {
                        str(span_id): dict(item)
                        for span_id, item in raw_spans.items()
                        if isinstance(item, dict)
                    }
                    for span_id, item in span_lookup.items():
                        item.setdefault("id_sp", span_id)
                else:
                    span_lookup = {
                        str(item["id_sp"]): item
                        for item in raw_spans
                        if isinstance(item, dict) and "id_sp" in item
                    }
                evidence: list[EvidenceItem] = []
                for ref in refs:
                    span = span_lookup.get(str(ref.get("id_sp")))
                    if span is not None:
                        evidence.append(
                            EvidenceItem(
                                text=str(span["text_sp"]),
                                is_float=False,
                                alignment_status="pending",
                            )
                        )
                if evidence:
                    query_id = f"{domain}:{dialogue['dial_id']}:{turn['turn_id']}"
                    queries_by_doc[(domain, doc_id)].append(
                        QueryRecord(
                            query_id=query_id,
                            question=normalize_text(str(turn.get("utterance") or "")),
                            answers=[AnswerAnnotation(annotation_id=query_id, evidence=evidence)],
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
                split=split,
                query_records=queries,
            )

            # Align grounding spans against the already-preserved span text.
            lookup: dict[str, list[str]] = defaultdict(list)
            for paragraph in document.paragraphs:
                lookup[normalize_text(paragraph.text)].append(paragraph.paragraph_id)
            for query in document.queries:
                for answer in query.answers:
                    for evidence in answer.evidence:
                        ids = lookup.get(normalize_text(evidence.text), [])
                        evidence.matched_paragraph_ids = ids
                        evidence.alignment_status = (
                            "matched_unique" if len(ids) == 1 else
                            "matched_ambiguous" if len(ids) > 1 else "unmatched"
                        )
            yield document
