from __future__ import annotations

import io
import json
import re
import difflib
import subprocess
import tarfile
import zipfile
from html import escape as html_escape
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pymupdf as fitz
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

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

QASA_DATASET_URL = (
    "https://github.com/lgresearch/QASA/raw/refs/heads/main/"
    "data/testset_answerable_1554_v1.1.json"
)

_QASA_ID_KEYS = ("arxiv_id", "paper_id", "article_id", "document_id", "paper")
_QASA_URL_KEYS = ("url", "paper_url", "arxiv_url", "ar5iv_url", "full_text_url")
_QUESTION_KEYS = ("question", "input")
_ANSWER_KEYS = ("answer", "free_form_answer", "output")
_EVIDENCE_KEYS = (
    "evidence",
    "evidential_info",
    "evidential_rationales",
    "evidential_rationale",
    "gold_evidence",
    "gold_ctxs",
    "gold_contexts",
)


@dataclass(frozen=True)
class QASAItem:
    question_id: str
    paper_id: str
    paper_url: str | None
    title: str
    question: str
    answer: str
    evidence_texts: tuple[str, ...]
    s2orc_id: str | None = None


def _first_direct(value: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in value and value[key] not in (None, ""):
            return value[key]
    return None


def _first_recursive(value: Any, keys: Iterable[str], *, max_depth: int = 3) -> Any:
    key_set = set(keys)
    if max_depth < 0:
        return None
    if isinstance(value, dict):
        for key in keys:
            if key in value and value[key] not in (None, ""):
                return value[key]
        for item in value.values():
            found = _first_recursive(item, key_set, max_depth=max_depth - 1)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value[:20]:
            found = _first_recursive(item, key_set, max_depth=max_depth - 1)
            if found not in (None, ""):
                return found
    return None


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "content", "paragraph", "evidence", "rationale"):
            if key in value and value[key] not in (None, ""):
                return _as_text(value[key])
    return ""


def _normalise_arxiv_id(value: Any) -> str:
    if isinstance(value, dict):
        nested = _first_direct(value, _QASA_ID_KEYS)
        if nested is not None and nested is not value:
            nested_id = _normalise_arxiv_id(nested)
            if nested_id:
                return nested_id
    text = _as_text(value)
    if not text:
        return ""
    text = text.strip()
    # Accept canonical arXiv URLs as well as bare IDs.
    match = re.search(r"arxiv\.org/(?:abs|html|pdf)/([^?#/]+)", text)
    if match:
        text = match.group(1)
    # arXiv version suffixes are retrieval metadata, not part of the canonical id.
    text = re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)
    return text


def _url_from_item(raw: dict[str, Any], paper_id: str) -> str | None:
    # The official QASA repository fetches full paper HTML from ar5iv using
    # the arXiv identifier. Prefer that path over the ``arxiv_url`` field,
    # which may point to an abstract/PDF endpoint rather than article HTML.
    if paper_id:
        return f"https://ar5iv.labs.arxiv.org/html/{paper_id}"

    value = _first_direct(raw, _QASA_URL_KEYS)
    if value is None:
        value = _first_recursive(raw, _QASA_URL_KEYS)
    url = _as_text(value)
    return url or None


def _extract_context_lookup(raw: dict[str, Any]) -> dict[str, str]:
    contexts = raw.get("ctxs") or raw.get("contexts") or raw.get("passages") or []
    lookup: dict[str, str] = {}
    if isinstance(contexts, dict):
        iterable = contexts.values()
    else:
        iterable = contexts if isinstance(contexts, list) else []
    for index, item in enumerate(iterable):
        if isinstance(item, dict):
            context_id = _as_text(item.get("id")) or str(index)
            text = _as_text(item.get("text") or item.get("content"))
        else:
            context_id = str(index)
            text = _as_text(item)
        if text:
            lookup[context_id] = text
            lookup[str(index)] = text
    return lookup



def _extract_evidential_info_paragraphs(value: Any) -> list[str]:
    """Extract source-paragraph text from QASA's evidential_info field.

    QASA stores evidence together with rationale annotations.  Prefer fields
    that identify the source paragraph/context and deliberately avoid treating
    generated rationale text as the source paragraph.
    """
    paragraph_keys = (
        "paragraph",
        "paragraph_text",
        "evidence_paragraph",
        "evidence_text",
        "context",
        "passage",
        "source_text",
        "text",
    )
    output: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            text = normalize_text(item)
            if text:
                output.append(text)
            return
        if isinstance(item, dict):
            # Prefer explicit source-text keys. Do not prefer rationale keys.
            for key in paragraph_keys:
                candidate = item.get(key)
                if candidate not in (None, ""):
                    text = _as_text(candidate)
                    if text:
                        output.append(text)
                        return

            # Some variants nest the source paragraph under an "evidence"
            # object/list. Recurse into likely source containers first.
            for key in ("evidence", "source", "grounding", "document"):
                if key in item:
                    visit(item[key])

            # Last resort: recurse only into nested containers. Do not treat
            # arbitrary string-valued fields as source evidence: QASA's
            # ``evidential_info`` records also contain annotation/rationale
            # strings that are not evidence paragraphs.
            for key, child in item.items():
                if str(key).casefold() in {
                    "rationale",
                    "evidential_rationale",
                    "evidential_rationales",
                    "composition",
                    "question",
                }:
                    continue
                if isinstance(child, (dict, list, tuple)):
                    visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)

    unique: list[str] = []
    seen: set[str] = set()
    for text in output:
        text = normalize_text(text)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _extract_evidence_texts(raw: dict[str, Any]) -> tuple[str, ...]:
    # QASA v1.1 uses ``evidential_info`` for the paragraph-level evidence
    # annotations. Handle that field explicitly because its nested structure
    # contains source paragraphs as well as generated rationales.
    if raw.get("evidential_info") not in (None, [], ""):
        # The official v1.1 field is authoritative. If it contains no source
        # paragraph text, do not fall through to generic evidence extraction,
        # which could mistake rationale strings for evidence.
        explicit = _extract_evidential_info_paragraphs(raw["evidential_info"])
        return tuple(explicit)

    value = _first_direct(raw, _EVIDENCE_KEYS)
    if value is None:
        value = _first_recursive(raw, _EVIDENCE_KEYS)
    if value is None:
        return ()

    lookup = _extract_context_lookup(raw)
    pieces: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            text = normalize_text(item)
            if text:
                # If the string is a context id, resolve it. Otherwise treat it as text.
                pieces.append(lookup.get(item, text))
            return
        if isinstance(item, dict):
            # Prefer an explicit text/content/evidence field.  If an id is present and
            # the object has no text, resolve it through the context lookup.
            text = _as_text(item.get("text") or item.get("content") or item.get("evidence"))
            if text:
                pieces.append(text)
                return
            item_id = _as_text(item.get("id") or item.get("context_id") or item.get("paragraph_id"))
            if item_id and item_id in lookup:
                pieces.append(lookup[item_id])
                return
            for child in item.values():
                visit(child)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return

    visit(value)

    unique: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        normalised = normalize_text(piece)
        if normalised and normalised not in seen:
            seen.add(normalised)
            unique.append(normalised)
    return tuple(unique)


def iter_qasa_items(raw_records: Any) -> list[QASAItem]:
    if isinstance(raw_records, dict):
        if isinstance(raw_records.get("data"), list):
            records = raw_records["data"]
        elif isinstance(raw_records.get("examples"), list):
            records = raw_records["examples"]
        else:
            # Some JSON releases use a mapping from paper/question IDs to records.
            records = list(raw_records.values())
    elif isinstance(raw_records, list):
        records = raw_records
    else:
        raise TypeError(f"Unsupported QASA JSON root type: {type(raw_records).__name__}")

    items: list[QASAItem] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            continue

        question = _as_text(_first_direct(raw, _QUESTION_KEYS))
        if not question:
            question = _as_text(_first_recursive(raw, _QUESTION_KEYS))
        if not question:
            continue

        paper_id_value = _first_direct(raw, _QASA_ID_KEYS)
        if paper_id_value is None:
            paper_id_value = _first_recursive(raw, _QASA_ID_KEYS)
        paper_id = _normalise_arxiv_id(paper_id_value)
        if not paper_id:
            raise ValueError(
                f"QASA record {index} has a question but no paper/arXiv identifier. "
                "Run scripts/probe_qasa.py against the downloaded file before proceeding."
            )

        question_id_value = _first_direct(raw, ("question_id", "qa_id", "qid", "id"))
        question_id = _as_text(question_id_value) or f"qasa:{paper_id}:q{index:04d}"
        title = _as_text(_first_direct(raw, ("title", "paper_title")))
        # QASA v1.1 does not expose the final answer as a top-level ``answer``
        # field in the released schema. Retrieval evaluation only needs the
        # annotated source evidence, so leave the answer blank when unavailable.
        answer = _as_text(_first_direct(raw, _ANSWER_KEYS))
        evidence_texts = _extract_evidence_texts(raw)
        s2orc_id_value = _first_direct(raw, ("s2orc_id", "s2orcId", "s2orc"))
        s2orc_id = _as_text(s2orc_id_value) or None

        items.append(
            QASAItem(
                question_id=question_id,
                paper_id=paper_id,
                paper_url=_url_from_item(raw, paper_id),
                title=title,
                question=question,
                answer=answer,
                evidence_texts=evidence_texts,
                s2orc_id=s2orc_id,
            )
        )
    return items


def load_qasa_items(json_path: Path) -> list[QASAItem]:
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return iter_qasa_items(raw)


def _section_heading(section: Tag) -> str:
    title = section.find(class_="ltx_title", recursive=False)
    if title is None:
        title = section.find(class_="ltx_title")
    return normalize_text(title.get_text(" ", strip=True)) if title else ""


def _section_id_number(section: Tag) -> str:
    value = section.get("id")
    return str(value) if value else "section"


def _top_level_sections(soup: BeautifulSoup) -> list[Tag]:
    sections: list[Tag] = []
    for section in soup.find_all("section"):
        section_id = str(section.get("id") or "")
        if re.fullmatch(r"S\d+", section_id):
            sections.append(section)
    return sections


_QASA_PARAGRAPH_ID_RE = re.compile(r"^(S\d+)\.(?:p|S)")


def _paragraph_tags(section: Tag, *, soup: BeautifulSoup | None = None) -> list[Tag]:
    """Return QASA/ar5iv paragraph nodes belonging to one top-level section.

    The official QASA helper does not rely on paragraph ancestry. It selects
    ``.ltx_para`` nodes whose ids start with ``S<number>.<p|S>``. In ar5iv,
    paragraphs in nested subsections therefore still belong to the parent
    top-level section. We follow that convention while retaining an ancestry
    fallback for simple synthetic test HTML without paragraph ids.
    """
    top_id = str(section.get("id") or "")
    if soup is None:
        soup = section if isinstance(section, BeautifulSoup) else section.find_parent()

    paragraphs: list[Tag] = []
    if soup is not None and top_id:
        candidates = soup.find_all(class_="ltx_para", id=_QASA_PARAGRAPH_ID_RE)
        for paragraph in candidates:
            pid = str(paragraph.get("id") or "")
            match = _QASA_PARAGRAPH_ID_RE.match(pid)
            if match and match.group(1) == top_id:
                paragraphs.append(paragraph)
        if paragraphs:
            return paragraphs

    # Fallback for minimal/synthetic HTML where paragraph ids are absent.
    for paragraph in section.find_all(class_="ltx_para"):
        ancestor = paragraph.find_parent(class_="ltx_para")
        if ancestor is not None:
            continue
        paragraphs.append(paragraph)
    return paragraphs


def _nearest_top_section(paragraph: Tag, top_sections: list[Tag]) -> Tag | None:
    for section in top_sections:
        if paragraph is section:
            return section
        if section in paragraph.parents:
            return section
    return None


def _append(buffer: list[str], text: str) -> CharacterSpan:
    start = sum(len(piece) for piece in buffer)
    buffer.append(text)
    return CharacterSpan(start=start, end=start + len(text))


def _append_document_block(buffer: list[str], text: str, separator: str) -> CharacterSpan:
    if buffer:
        buffer.append(separator)
    return _append(buffer, text)


def _parse_qasa_html(
    *,
    paper_id: str,
    html: str,
    split: str,
    title_hint: str = "",
    config: Any,
    query_items: list[QASAItem],
) -> DocumentRecord:
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", class_="ltx_title_document") or soup.find("title")
    title = normalize_text(title_tag.get_text(" ", strip=True)) if title_tag else title_hint
    abstract_tag = soup.find("div", class_="ltx_abstract")
    abstract = normalize_text(abstract_tag.get_text(" ", strip=True)) if abstract_tag else ""

    buffer: list[str] = []
    sections: list[SectionRecord] = []
    paragraphs: list[ParagraphRecord] = []

    def add_block(text: str, heading: str, *, source_index: int, is_abstract: bool) -> None:
        if not text and not heading:
            return
        section_id = f"{paper_id}:section:{len(sections):04d}"
        if buffer:
            buffer.append(config.document.section_separator)
        section_start = sum(len(piece) for piece in buffer)
        if heading:
            buffer.append(heading)
            buffer.append(config.document.heading_separator)

        paragraph_ids: list[str] = []
        if text:
            paragraph_texts = [normalize_text(part) for part in text if normalize_text(part)]
            for local_index, paragraph_text in enumerate(paragraph_texts):
                if paragraph_ids:
                    buffer.append(config.document.paragraph_separator)
                span = _append(buffer, paragraph_text)
                paragraph_id = f"{paper_id}:paragraph:{len(paragraphs):05d}"
                paragraphs.append(
                    ParagraphRecord(
                        paragraph_id=paragraph_id,
                        section_id=section_id,
                        source_section_index=source_index,
                        source_paragraph_index=local_index,
                        text=paragraph_text,
                        span=span,
                    )
                )
                paragraph_ids.append(paragraph_id)

        section_end = sum(len(piece) for piece in buffer)
        if paragraph_ids:
            sections.append(
                SectionRecord(
                    section_id=section_id,
                    source_section_index=source_index,
                    heading=heading,
                    top_level_heading=heading,
                    hierarchy=[heading] if heading else [],
                    span=CharacterSpan(start=section_start, end=section_end),
                    paragraph_ids=paragraph_ids,
                    is_abstract=is_abstract,
                )
            )

    if config.document.include_title and title:
        # Keep the title inside the abstract block when an abstract is present,
        # matching the QASPER reconstruction convention.
        abstract_paragraphs = []
        if abstract:
            abstract_paragraphs.append(title)
        if abstract:
            abstract_paragraphs.append(abstract)
        else:
            abstract_paragraphs.append(title)
        add_block(abstract_paragraphs, config.document.abstract_heading if abstract else "Title", source_index=-1, is_abstract=bool(abstract))
    elif abstract and config.document.include_abstract:
        add_block([abstract], config.document.abstract_heading, source_index=-1, is_abstract=True)

    top_sections = _top_level_sections(soup)
    for source_index, top_section in enumerate(top_sections):
        heading = _section_heading(top_section)
        if config.document.remove_reference_sections:
            candidate = heading.casefold().rstrip(":")
            if candidate in {normalize_text(item).casefold().rstrip(":") for item in config.document.reference_headings}:
                continue

        paragraph_texts: list[str] = []
        for paragraph in _paragraph_tags(top_section, soup=soup):
            text = normalize_text(paragraph.get_text(" ", strip=True))
            if text:
                paragraph_texts.append(text)
        add_block(paragraph_texts, heading, source_index=source_index, is_abstract=False)

    document = DocumentRecord(
        document_id=paper_id,
        split=split,
        title=title,
        abstract=abstract,
        text="".join(buffer),
        sections=sections,
        paragraphs=paragraphs,
        queries=[
            QueryRecord(
                # QASA question IDs restart at 0 for each paper. Retrieval
                # validation and ranking operate over the selected corpus, so
                # query IDs must be globally unique across documents.
                query_id=f"{paper_id}:{item.question_id}",
                question=item.question,
                answers=[
                    AnswerAnnotation(
                        annotation_id=item.question_id,
                        free_form_answer=item.answer,
                        evidence=[
                            EvidenceItem(
                                text=text,
                                is_float=False,
                                alignment_status="pending",
                            )
                            for text in item.evidence_texts
                        ],
                    )
                ],
            )
            for item in query_items
        ],
    )

    # QASA gold evidence is defined over body ``ltx_para`` nodes. Keep title/abstract
    # available in the retrieval document, but never use them as evidence targets.
    section_by_id = {section.section_id: section for section in document.sections}
    body_paragraphs = [
        paragraph
        for paragraph in document.paragraphs
        if not section_by_id[paragraph.section_id].is_abstract
    ]

    paragraph_lookup: dict[str, list[str]] = defaultdict(list)
    compact_lookup: dict[str, list[str]] = defaultdict(list)
    compact_paragraphs: list[tuple[str, str, int, int]] = []
    compact_document_parts: list[str] = []
    compact_offset = 0

    for paragraph in body_paragraphs:
        normalized = normalize_text(paragraph.text)
        paragraph_lookup[normalized].append(paragraph.paragraph_id)

        compact = re.sub(r"\s+", "", normalized)
        if compact:
            compact_lookup[compact].append(paragraph.paragraph_id)
            start = compact_offset
            end = start + len(compact)
            compact_paragraphs.append((paragraph.paragraph_id, compact, start, end))
            compact_document_parts.append(compact)
            compact_offset = end

    compact_document = "".join(compact_document_parts)

    def _token_projection(value: str) -> list[tuple[str, int, int]]:
        """Project text to comparison tokens while retaining source character offsets.

        QASA evidence can differ from a reconstructed source paragraph because of
        citation formatting, superscript footnotes, punctuation, and whitespace.
        Splitting alphanumeric boundaries lets strings such as ``Law222Strathern``
        align with ``Law Strathern`` while keeping a mapping back to document offsets.
        """
        value = value.replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", " ", value)
        token_matches = re.finditer(r"[^\W_]+(?:['’\-][^\W_]+)*", value, flags=re.UNICODE)
        return [
            (match.group(0).casefold(), match.start(), match.end())
            for match in token_matches
        ]

    doc_tokens = _token_projection(document.text)
    doc_token_values = [token for token, _, _ in doc_tokens]
    doc_ngram_index: dict[int, defaultdict[tuple[str, ...], list[int]]] = {}
    for n in (5, 6, 8):
        index: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
        if len(doc_token_values) >= n:
            for i in range(len(doc_token_values) - n + 1):
                index[tuple(doc_token_values[i : i + n])].append(i)
        doc_ngram_index[n] = index

    def _comparison_tokens(value: str) -> list[str]:
        """Return prose-oriented tokens for robust paragraph-level matching.

        QASA evidence and reconstructed PDF/TeX paragraphs can differ in the
        rendering of equations, Greek symbols, TeX commands, and superscript
        footnotes. For paragraph identification we keep ordinary lexical
        content and discard tokens that are mostly mathematical/control syntax.
        """
        value = value.replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", " ", value)
        tokens = [
            match.group(0).casefold()
            for match in re.finditer(
                r"[^\W_]+(?:['’\-][^\W_]+)*", value, flags=re.UNICODE
            )
        ]
        excluded = {
            "mathbf", "mathrm", "mathbb", "mathcal", "operatorname",
            "pi", "theta", "phi", "psi", "tau", "gamma", "lambda",
            "delta", "mu", "epsilon", "sigma", "alpha", "beta",
            "hat", "argmax", "min", "max", "clip",
        }
        return [
            token for token in tokens
            if len(token) >= 2
            and not re.fullmatch(r"\d+(?:\.\d+)?", token)
            and token not in excluded
        ]

    paragraph_comparison_tokens = [
        _comparison_tokens(paragraph.text) for paragraph in body_paragraphs
    ]

    def _fuzzy_single_paragraph(text: str) -> tuple[int, float] | None:
        """Prefer a high-confidence single paragraph before span expansion.

        This is intentionally paragraph-local. It prevents a near-perfect match
        to one evidence paragraph from being expanded to an adjacent paragraph
        merely because a wider fuzzy window scores marginally better.
        """
        evidence_tokens = _comparison_tokens(text)
        if len(evidence_tokens) < 8:
            return None

        evidence_counts = defaultdict(int)
        for token in evidence_tokens:
            evidence_counts[token] += 1
        evidence_anchors = {
            tuple(evidence_tokens[i : i + 5])
            for i in range(len(evidence_tokens) - 4)
        }

        scored: list[tuple[float, float, float, int]] = []
        for index, candidate_tokens in enumerate(paragraph_comparison_tokens):
            if not candidate_tokens or not any(
                tuple(candidate_tokens[i : i + 5]) in evidence_anchors
                for i in range(len(candidate_tokens) - 4)
            ):
                continue
            ratio = difflib.SequenceMatcher(
                None, evidence_tokens, candidate_tokens, autojunk=False
            ).ratio()
            candidate_counts = defaultdict(int)
            for token in candidate_tokens:
                candidate_counts[token] += 1
            shared = sum(
                min(count, candidate_counts[token])
                for token, count in evidence_counts.items()
            ) / max(1, len(evidence_tokens))
            length_ratio = len(candidate_tokens) / max(1, len(evidence_tokens))
            scored.append((ratio, shared, length_ratio, index))

        if not scored:
            return None
        scored.sort(reverse=True)
        best = scored[0]
        second = scored[1] if len(scored) > 1 else (0.0, 0.0, 0.0, -1)
        ratio, shared, length_ratio, index = best
        margin = ratio - second[0]
        if (
            ratio >= 0.82
            and shared >= 0.75
            and 0.45 <= length_ratio <= 1.60
            and (margin >= 0.035 or ratio >= 0.90)
        ):
            return index, ratio
        return None

    def _scattered_anchor_matches(text: str) -> list[str]:
        """Recover evidence split across non-contiguous extracted fragments.

        PDF reconstruction can interrupt a source paragraph with a figure,
        footnote, or equation block. We therefore use exact 8-token lexical
        anchors to identify a small ordered set of paragraph fragments. This is
        a last resort and requires substantial evidence-token coverage.
        """
        evidence_tokens = _comparison_tokens(text)
        n = 8
        if len(evidence_tokens) < n:
            return []

        index: defaultdict[tuple[str, ...], set[int]] = defaultdict(set)
        paragraph_tokens = paragraph_comparison_tokens
        for paragraph_index, tokens in enumerate(paragraph_tokens):
            for start in range(len(tokens) - n + 1):
                index[tuple(tokens[start:start + n])].add(paragraph_index)

        positions: defaultdict[int, set[int]] = defaultdict(set)
        for evidence_start in range(len(evidence_tokens) - n + 1):
            anchor = tuple(evidence_tokens[evidence_start:evidence_start + n])
            for paragraph_index in index.get(anchor, ()):
                positions[paragraph_index].add(evidence_start)

        candidates = [
            (paragraph_index, matched_positions)
            for paragraph_index, matched_positions in positions.items()
            if len(matched_positions) >= 3
        ]
        if not candidates:
            return []

        total_positions = max(1, len(evidence_tokens) - n + 1)
        selected: list[tuple[int, set[int]]] = []
        covered: set[int] = set()
        for paragraph_index, matched_positions in sorted(
            candidates, key=lambda item: len(item[1]), reverse=True
        ):
            additional = matched_positions - covered
            if len(additional) < 3:
                continue
            fraction = len(additional) / total_positions
            if not selected or fraction >= 0.025:
                selected.append((paragraph_index, additional))
                covered.update(additional)
            if len(selected) >= 3:
                break

        coverage = len(covered) / total_positions
        if coverage < 0.50:
            return []
        selected.sort(key=lambda item: min(item[1]))
        selected_indices = [index for index, _ in selected]
        # A contiguous span is handled by the established fuzzy-span matcher.
        # Scattered matching is reserved for extraction artifacts that visibly
        # interrupt one logical evidence paragraph with unrelated blocks.
        if len(selected_indices) < 2 or all(
            selected_indices[i + 1] == selected_indices[i] + 1
            for i in range(len(selected_indices) - 1)
        ):
            return []
        return [body_paragraphs[index].paragraph_id for index, _ in selected]

    def _candidate_windows(evidence_tokens: list[str]) -> list[tuple[int, int, float]]:
        """Find likely document windows using exact token n-gram anchors.

        This remains lexical, not semantic: every candidate must contain an
        exact multi-token anchor copied from the evidence. The fuzzy step only
        scores the local window around such an anchor.
        """
        if not evidence_tokens or not doc_token_values:
            return []

        anchor_lengths = [n for n in (8, 6, 5) if len(evidence_tokens) >= n]
        if not anchor_lengths:
            return []

        candidates: list[tuple[int, int, float]] = []
        for anchor_len in anchor_lengths:
            evidence_positions = list(range(0, len(evidence_tokens) - anchor_len + 1))
            # Prefer anchors with uncommon document frequency.
            index = doc_ngram_index.get(anchor_len, {})
            frequency: list[tuple[int, int]] = []
            for pos in evidence_positions:
                anchor = tuple(evidence_tokens[pos : pos + anchor_len])
                count = len(index.get(anchor, []))
                if count:
                    frequency.append((count, pos))
            frequency.sort()
            for count, evidence_anchor_start in frequency[:4]:
                anchor = tuple(evidence_tokens[evidence_anchor_start : evidence_anchor_start + anchor_len])
                for doc_anchor_start in index.get(anchor, []):
                    expected_start = max(0, doc_anchor_start - evidence_anchor_start)
                    evidence_len = len(evidence_tokens)
                    min_len = max(anchor_len, int(evidence_len * 0.55))
                    max_len = max(min_len, int(evidence_len * 1.45))
                    for window_len in {min_len, evidence_len, max_len}:
                        start = expected_start
                        end = min(len(doc_token_values), start + window_len)
                        if end - start >= anchor_len and start <= doc_anchor_start <= end - anchor_len:
                            candidates.append((start, end, 1.0))
                    if len(candidates) >= 20:
                        return candidates
            if candidates:
                return candidates
        return candidates

    def _fuzzy_evidence_span(text: str) -> tuple[int, int, float] | None:
        evidence_tokens = [token for token, _, _ in _token_projection(text)]
        if len(evidence_tokens) < 8:
            return None
        best: tuple[float, int, int] | None = None
        seen_windows: set[tuple[int, int]] = set()
        for start_token, end_token, _ in _candidate_windows(evidence_tokens):
            key = (start_token, end_token)
            if key in seen_windows:
                continue
            seen_windows.add(key)
            candidate_tokens = doc_token_values[start_token:end_token]
            if not candidate_tokens:
                continue
            score = difflib.SequenceMatcher(
                None, evidence_tokens, candidate_tokens, autojunk=False
            ).ratio()
            evidence_len = len(evidence_tokens)
            candidate_len = len(candidate_tokens)
            if candidate_len < max(4, int(evidence_len * 0.45)):
                continue
            if candidate_len > int(evidence_len * 1.60):
                continue
            if best is None or score > best[0]:
                best = (score, start_token, end_token)

        if best is None or best[0] < 0.80:
            return None

        score, start_token, end_token = best
        # Recover source character offsets from the token projection.
        start_char = doc_tokens[start_token][1]
        end_char = doc_tokens[end_token - 1][2]
        return start_char, end_char, score

    def _paragraph_ids_for_char_span(start: int, end: int) -> list[str]:
        return [
            paragraph.paragraph_id
            for paragraph in body_paragraphs
            if paragraph.span.start < end and paragraph.span.end > start
        ]

    def _match_evidence_paragraphs(text: str) -> tuple[list[str], str]:
        """Align evidence text exactly first, then with conservative lexical span recovery.

        The fallback is deliberately not semantic. It requires exact multi-token anchors
        from the evidence and a high local sequence-similarity score. This is intended to
        recover formatting/version drift without inventing unrelated relevance labels.
        """
        key = normalize_text(text)
        if not key:
            return [], "empty"

        matches = list(paragraph_lookup.get(key, []))
        if matches:
            if len(matches) == 1:
                return matches, "matched_unique"
            # The same gold evidence text occurs in multiple body paragraphs.
            # We cannot identify the intended paragraph from text alone, so do not
            # silently turn ambiguity into multiple relevant passages.
            return matches, "matched_ambiguous"

        compact = re.sub(r"\s+", "", key)
        if compact:
            matches = list(compact_lookup.get(compact, []))
            if matches:
                if len(matches) == 1:
                    return matches, "matched_unique"
                return matches, "matched_ambiguous"

            start = compact_document.find(compact)
            if start >= 0:
                end = start + len(compact)
                overlapping = [
                    paragraph_id
                    for paragraph_id, _, paragraph_start, paragraph_end in compact_paragraphs
                    if paragraph_start < end and paragraph_end > start
                ]
                if overlapping:
                    return overlapping, "matched_unique" if len(overlapping) == 1 else "matched_span"

        # Prefer a high-confidence single-paragraph match. This prevents a
        # paragraph-local evidence item from being expanded to an adjacent
        # paragraph simply because a broader fuzzy window scores slightly higher.
        single = _fuzzy_single_paragraph(key)
        if single is not None:
            index, _ = single
            return [body_paragraphs[index].paragraph_id], "matched_fuzzy_unique"

        scattered = _scattered_anchor_matches(key)
        if scattered:
            return scattered, "matched_fuzzy_scattered"

        fuzzy_span = _fuzzy_evidence_span(key)
        if fuzzy_span is not None:
            start, end, score = fuzzy_span
            matches = _paragraph_ids_for_char_span(start, end)
            if matches:
                status = "matched_fuzzy_unique" if len(matches) == 1 else "matched_fuzzy_span"
                return matches, status

        return [], "unmatched"

    for query in document.queries:
        for answer in query.answers:
            for evidence in answer.evidence:
                matches, status = _match_evidence_paragraphs(evidence.text)
                evidence.matched_paragraph_ids = list(matches)
                evidence.alignment_status = status

    return document




def _looks_like_qasa_pdf(data: bytes) -> bool:
    if not data or len(data) < 10_000:
        return False
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        ok = pdf.page_count >= 2
        pdf.close()
        return ok
    except Exception:
        return False


def _is_pdf_caption(text: str) -> bool:
    """Return whether a text block is a figure/table-style caption."""
    text = normalize_text(text)
    return bool(
        re.match(
            r"^(?:Figure|Fig\.|Table|Algorithm|Appendix\s+(?:Figure|Table))\s+\d+[\s:.)-]",
            text,
            flags=re.IGNORECASE,
        )
    )


def _rect_contains_point(
    rect: tuple[float, float, float, float], x: float, y: float
) -> bool:
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _rect_expand(
    rect: tuple[float, float, float, float], *, dx: float = 0.0, dy: float = 0.0
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = rect
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def _pdf_graphic_regions(
    page: Any,
    raw_blocks: list[tuple[float, float, float, float, str]],
) -> list[tuple[float, float, float, float]]:
    """Find likely figure/table regions to exclude from the body stream."""
    regions: list[tuple[float, float, float, float]] = []

    try:
        for image in page.get_images(full=True):
            try:
                rects = page.get_image_rects(image[0])
            except Exception:
                continue
            for rect in rects:
                regions.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))
    except Exception:
        pass

    captions = [block for block in raw_blocks if _is_pdf_caption(block[4])]
    for caption in captions:
        cx0, cy0, cx1, cy1, _ = caption
        best = None
        best_distance = float("inf")
        for region in regions:
            rx0, ry0, rx1, ry1 = region
            if ry1 > cy0 + 20:
                continue
            x_overlap = max(0.0, min(cx1, rx1) - max(cx0, rx0))
            if x_overlap <= 0:
                continue
            distance = cy0 - ry1
            if distance < best_distance:
                best_distance = distance
                best = region
        if best is not None and best_distance < 180:
            regions.append(
                (
                    min(best[0], cx0),
                    min(best[1], cy0),
                    max(best[2], cx1),
                    max(best[3], cy1),
                )
            )
        else:
            # Conservative fallback for vector-only figures/tables: exclude a
            # centered area immediately above the caption.
            page_width = float(page.rect.width)
            width = min(page_width * 0.85, max(cx1 - cx0, page_width * 0.45))
            x0 = max(0.0, (page_width - width) / 2.0)
            x1 = min(page_width, x0 + width)
            regions.append((x0, max(0.0, cy0 - 220.0), x1, cy1))
    return regions


def _pdf_blocks(data: bytes) -> list[tuple[int, float, float, str]]:
    """Extract ordered body-oriented text blocks from a PDF.

    QASA's HTML extractor operates on body paragraphs and excludes figure/table
    content. PDF extraction does not have that semantic boundary, so we remove
    repeated headers/footers, captions, and text located in detected graphic
    regions before reconstructing paragraph units.
    """
    pdf = fitz.open(stream=data, filetype="pdf")
    blocks: list[tuple[int, float, float, str]] = []
    try:
        repeated: dict[str, int] = defaultdict(int)
        page_records: list[
            tuple[int, list[tuple[float, float, float, float, str]], float]
        ] = []

        for page_index, page in enumerate(pdf):
            page_height = float(page.rect.height)
            raw_page: list[tuple[float, float, float, float, str]] = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7:
                    continue
                x0, y0, x1, y1, text, _, block_type = block[:7]
                if block_type != 0:
                    continue
                text = normalize_text(str(text))
                if not text or re.fullmatch(r"\d+", text):
                    continue
                raw_page.append((float(x0), float(y0), float(x1), float(y1), text))
                if y0 < page_height * 0.09 or y1 > page_height * 0.91:
                    repeated[text] += 1
            page_records.append((page_index, raw_page, page_height))

        for page_index, raw_page, page_height in page_records:
            graphic_regions = _pdf_graphic_regions(pdf[page_index], raw_page)
            for x0, y0, x1, y1, text in raw_page:
                if (y0 < page_height * 0.09 or y1 > page_height * 0.91) and repeated[text] >= 2:
                    continue
                if _is_pdf_caption(text):
                    continue
                center_x = (x0 + x1) / 2.0
                center_y = (y0 + y1) / 2.0
                if any(
                    _rect_contains_point(_rect_expand(region, dx=6.0, dy=6.0), center_x, center_y)
                    for region in graphic_regions
                ):
                    continue
                blocks.append((page_index, y0, x0, text))
    finally:
        pdf.close()
    return blocks


def _is_pdf_continuation(previous: str, current: str) -> bool:
    """Join blocks when the second block is clearly a continuation."""
    previous = previous.rstrip()
    current = current.lstrip()
    if not previous or not current:
        return False
    if previous.endswith("-"):
        return True
    if re.search(r"[.!?]$", previous):
        return False
    return current[0] in "([{,;:)—–-" or current[0].islower()


def _merge_pdf_blocks(blocks: list[tuple[int, float, float, str]]) -> list[str]:
    """Merge page/figure-separated fragments into logical paragraph units."""
    paragraphs: list[str] = []
    current: str | None = None
    for _, _, _, text in blocks:
        text = normalize_text(text)
        if not text:
            continue
        if current is not None and _is_pdf_continuation(current, text):
            if current.endswith("-"):
                current = current[:-1] + text
            else:
                current = f"{current} {text}"
            continue
        if current is not None:
            paragraphs.append(current)
        current = text
    if current is not None:
        paragraphs.append(current)
    return paragraphs


def _is_top_level_pdf_heading(text: str) -> bool:
    text = normalize_text(text)
    if not text or len(text) > 140 or "figure" in text.casefold() or "table" in text.casefold():
        return False
    if re.match(r"^\d+\s+[A-Za-z][^.!?]{1,100}$", text):
        return True
    if re.match(r"^[IVX]+\s+[A-Za-z][^.!?]{1,100}$", text):
        return True
    common = {
        "introduction", "background", "related work", "method", "methods",
        "methodology", "experiments", "results", "evaluation", "discussion",
        "conclusion", "conclusions", "analysis", "findings", "limitations",
        "future work", "acknowledgements", "acknowledgments",
    }
    return text.casefold().rstrip(":") in common


def _pdf_to_qasa_html(data: bytes, *, paper_id: str, title_hint: str) -> str:
    """Convert a versioned PDF into the minimal QASA HTML structure."""
    blocks = _pdf_blocks(data)
    if not blocks:
        raise ValueError("PDF contained no extractable text blocks")

    headings: list[tuple[str, list[str]]] = []
    preamble: list[str] = []
    current_heading: str | None = None
    current_paragraphs: list[str] = []

    merged_blocks = _merge_pdf_blocks(blocks)
    for text in merged_blocks:
        if _is_top_level_pdf_heading(text):
            if current_heading is not None:
                headings.append((current_heading, current_paragraphs))
            current_heading = text
            current_paragraphs = []
        else:
            if current_heading is None:
                preamble.append(text)
            else:
                current_paragraphs.append(text)
    if current_heading is not None:
        headings.append((current_heading, current_paragraphs))

    # Keep a bounded preamble as the abstract/title block. This captures the
    # front matter without letting author/affiliation boilerplate swamp the body.
    abstract_text = " ".join(preamble[:12])
    if not abstract_text:
        abstract_text = title_hint

    pieces = [
        '<html><head><meta name="qasa-pdf-source" content="1"></head><body>',
        f'<h1 class="ltx_title_document">{html_escape(title_hint or paper_id)}</h1>',
    ]
    if abstract_text:
        pieces.append(f'<div class="ltx_abstract">{html_escape(abstract_text)}</div>')

    for index, (heading, paragraphs) in enumerate(headings, start=1):
        pieces.append(f'<section id="S{index}"><h2 class="ltx_title">{html_escape(heading)}</h2>')
        paragraph_index = 0
        for paragraph in paragraphs:
            text = normalize_text(paragraph)
            if not text:
                continue
            pieces.append(
                f'<div class="ltx_para" id="S{index}.p{paragraph_index}">'
                f'<p>{html_escape(text)}</p></div>'
            )
            paragraph_index += 1
        pieces.append("</section>")
    pieces.append("</body></html>")
    return "".join(pieces)


def _pdf_evidence_score(data: bytes, evidence_texts: Iterable[str]) -> tuple[int, int, str]:
    """Score a versioned PDF and return matched count plus reconstructed HTML."""
    texts = []
    seen: set[str] = set()
    for evidence in evidence_texts:
        normalized = normalize_text(evidence)
        if normalized and normalized not in seen:
            seen.add(normalized)
            texts.append(normalized)
    if not texts or not _looks_like_qasa_pdf(data):
        return (0, len(texts), "")

    blocks = _merge_pdf_blocks(_pdf_blocks(data))
    compact_blocks = {re.sub(r"\s+", "", text) for text in blocks if text}
    matched = 0
    for evidence in texts:
        compact = re.sub(r"\s+", "", evidence)
        if compact and compact in compact_blocks:
            matched += 1
    return matched, len(texts), ""

def _looks_like_qasa_full_text(html: str) -> bool:
    """Check whether cached/downloaded HTML contains a usable QASA body."""
    if not html:
        return False
    if (
        'name="qasa-pdf-source"' not in html
        and 'name="qasa-source"' not in html
        and len(html) < 1000
    ):
        return False
    soup = BeautifulSoup(html, "html.parser")
    top_sections = soup.find_all("section", attrs={"id": re.compile(r"^S\d+$")})
    paragraphs = soup.find_all(class_="ltx_para", id=_QASA_PARAGRAPH_ID_RE)
    return bool(top_sections and paragraphs)



def _html_evidence_score(html: str, evidence_texts: Iterable[str]) -> tuple[int, int]:
    """Score a full-text HTML candidate against QASA evidence annotations.

    Matching mirrors the document-alignment routine: exact normalized paragraph
    equality first, then exact whitespace-insensitive substring matching across
    the reconstructed paragraph stream. This is only a source-version consistency
    check and never invents semantic relevance labels.
    """
    texts = []
    seen = set()
    for evidence in evidence_texts:
        normalized = normalize_text(evidence)
        if normalized and normalized not in seen:
            seen.add(normalized)
            texts.append(normalized)
    if not texts or not _looks_like_qasa_full_text(html):
        return (0, len(texts))

    soup = BeautifulSoup(html, "html.parser")
    normalized_set: set[str] = set()
    compact_paragraphs: set[str] = set()
    for paragraph in soup.find_all(class_="ltx_para", id=_QASA_PARAGRAPH_ID_RE):
        text = normalize_text(paragraph.get_text(" ", strip=True))
        if text:
            normalized_set.add(text)
            compact_paragraphs.add(re.sub(r"\s+", "", text))

    matched = 0
    for evidence in texts:
        if evidence in normalized_set:
            matched += 1
            continue
        compact_evidence = re.sub(r"\s+", "", evidence)
        if compact_evidence and compact_evidence in compact_paragraphs:
            matched += 1
    return matched, len(texts)



def _html_fuzzy_evidence_score(
    html: str, evidence_texts: Iterable[str]
) -> tuple[int, int]:
    """Estimate evidence coverage under conservative lexical rendering drift.

    This is used only to choose among alternate representations of the same
    paper (current ar5iv, historical ar5iv/source, PDF reconstruction). It is
    intentionally weaker than the final paragraph alignment: an evidence item
    counts when a single extracted paragraph has a strong token-level match,
    contains an exact five-token anchor, and has compatible length.
    """
    texts: list[str] = []
    seen: set[str] = set()
    for evidence in evidence_texts:
        normalized = normalize_text(evidence)
        if normalized and normalized not in seen:
            seen.add(normalized)
            texts.append(normalized)
    if not texts or not _looks_like_qasa_full_text(html):
        return (0, len(texts))

    def tokens(value: str) -> list[str]:
        value = value.replace("’", "'").replace("“", '"').replace("”", '"')
        value = re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", " ", value)
        raw = [
            match.group(0).casefold()
            for match in re.finditer(
                r"[^\W_]+(?:['’\-][^\W_]+)*", value, flags=re.UNICODE
            )
        ]
        excluded = {
            "mathbf", "mathrm", "mathbb", "mathcal", "operatorname",
            "pi", "theta", "phi", "psi", "tau", "gamma", "lambda",
            "delta", "mu", "epsilon", "sigma", "alpha", "beta",
            "hat", "argmax", "min", "max", "clip",
        }
        return [
            token for token in raw
            if len(token) >= 2
            and not re.fullmatch(r"\d+(?:\.\d+)?", token)
            and token not in excluded
        ]

    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [
        normalize_text(paragraph.get_text(" ", strip=True))
        for paragraph in soup.find_all(class_="ltx_para", id=_QASA_PARAGRAPH_ID_RE)
    ]
    paragraph_tokens = [tokens(text) for text in paragraphs if text]
    matched = 0

    for evidence in texts:
        evidence_tokens = tokens(evidence)
        if len(evidence_tokens) < 8:
            continue
        anchors = {
            tuple(evidence_tokens[i : i + 5])
            for i in range(len(evidence_tokens) - 4)
        }
        best = 0.0
        for candidate in paragraph_tokens:
            if len(candidate) < 5:
                continue
            if not any(
                tuple(candidate[i : i + 5]) in anchors
                for i in range(len(candidate) - 4)
            ):
                continue
            ratio = difflib.SequenceMatcher(
                None, evidence_tokens, candidate, autojunk=False
            ).ratio()
            length_ratio = len(candidate) / max(1, len(evidence_tokens))
            if 0.45 <= length_ratio <= 1.60:
                best = max(best, ratio)
        if best >= 0.82:
            matched += 1

    return matched, len(texts)


def _candidate_selection_key(
    candidate: tuple[int | None, str, tuple[int, int], str],
    evidence_texts: Iterable[str],
) -> tuple[int, int, int, int, int]:
    """Rank alternate paper representations by recoverable evidence coverage.

    Lexical coverage is considered before exact-string coverage because ar5iv
    often changes punctuation, citation markers, footnote formatting, and math
    rendering without changing the underlying evidence. Exact coverage remains
    a secondary tie-breaker and version/current-page preference is retained
    after it.
    """
    version, _, exact_score, html = candidate
    fuzzy_count, _ = _html_fuzzy_evidence_score(html, evidence_texts)
    return (
        fuzzy_count,
        exact_score[0],
        exact_score[1],
        1 if version is None else 0,
        version if version is not None else -1,
    )

def _safe_archive_member(name: str) -> str | None:
    """Return a safe relative archive path, or ``None`` for unsafe entries."""
    name = name.replace("\\", "/")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return None
    return name.lstrip("./") or None


def _extract_source_files(data: bytes) -> dict[str, str]:
    """Extract text source files from an arXiv e-print archive."""
    files: dict[str, str] = {}

    def add(name: str, payload: bytes) -> None:
        safe = _safe_archive_member(name)
        if not safe:
            return
        suffix = Path(safe).suffix.casefold()
        if suffix not in {".tex", ".ltx", ".sty", ".cls", ".bib", ".cfg"}:
            return
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("latin-1", errors="ignore")
        files[safe] = text

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    add(member.name, extracted.read())
            if files:
                return files
    except (tarfile.TarError, OSError):
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                add(info.filename, archive.read(info))
            if files:
                return files
    except (zipfile.BadZipFile, OSError):
        pass

    try:
        raw = data
        if data[:2] == b"\x1f\x8b":
            import gzip
            raw = gzip.decompress(data)
        add("main.tex", raw)
    except Exception:
        pass
    return files


def _strip_latex_comments(text: str) -> str:
    """Remove TeX comments while preserving escaped percent signs."""
    output: list[str] = []
    for line in text.splitlines():
        escaped = False
        cut = None
        for index, char in enumerate(line):
            if char == "%" and not escaped:
                cut = index
                break
            if char == "\\":
                escaped = not escaped
            else:
                escaped = False
        output.append(line if cut is None else line[:cut])
    return "\n".join(output)


def _find_main_tex(files: dict[str, str]) -> tuple[str, str] | None:
    candidates: list[tuple[int, str, str]] = []
    for name, text in files.items():
        if Path(name).suffix.casefold() not in {".tex", ".ltx"}:
            continue
        if "\\begin{document}" not in text:
            continue
        score = len(text)
        if Path(name).stem.casefold() in {"main", "paper", "article", "manuscript"}:
            score += 1_000_000
        candidates.append((score, name, text))
    if not candidates:
        return None
    _, name, text = max(candidates, key=lambda item: item[0])
    return name, text


def _resolve_tex_inputs(
    text: str,
    *,
    files: dict[str, str],
    current_path: str,
    seen: set[str] | None = None,
    depth: int = 0,
) -> str:
    """Inline local TeX ``input/include/subfile`` files without requiring TeX."""
    if depth > 20:
        return text
    seen = set() if seen is None else seen
    current_dir = Path(current_path).parent
    pattern = re.compile(r"\\(?:input|include|subfile)\s*(?:\{([^{}]+)\}|([^\s%]+))")

    def replace(match: re.Match[str]) -> str:
        requested = (match.group(1) or match.group(2) or "").strip()
        candidates = [
            str((current_dir / requested).as_posix()),
            str((current_dir / f"{requested}.tex").as_posix()),
            str((current_dir / f"{requested}.ltx").as_posix()),
        ]
        target = next((candidate for candidate in candidates if candidate in files), None)
        if target is None or target in seen:
            return ""
        seen.add(target)
        return "\n" + _resolve_tex_inputs(
            files[target], files=files, current_path=target, seen=seen, depth=depth + 1
        ) + "\n"

    return pattern.sub(replace, text)


def _load_bib_authors(files: dict[str, str]) -> dict[str, str]:
    """Build cite-key -> compact author/year text for source-text matching."""
    lookup: dict[str, str] = {}
    for name, text in files.items():
        if Path(name).suffix.casefold() != ".bib":
            continue
        for entry in re.finditer(
            r"@[^\{]+\{\s*([^,]+),(?P<body>.*?)\n\}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            key = entry.group(1).strip()
            body = entry.group("body")
            author_match = re.search(r"\bauthor\s*=\s*\{(.*?)\}", body, flags=re.I | re.S)
            year_match = re.search(r"\byear\s*=\s*\{(\d{4})\}", body, flags=re.I)
            if not author_match:
                continue
            authors = re.sub(r"\s+", " ", author_match.group(1)).strip()
            year = year_match.group(1) if year_match else ""
            author_parts = re.split(r"\s+and\s+", authors, flags=re.I)
            first = author_parts[0].split(",", 1)[0].strip() or author_parts[0].strip()
            surname = first.split()[-1] if first else ""
            if not surname:
                continue
            if len(author_parts) > 1 and year:
                rendered = f"{surname} et al., {year}"
            elif year:
                rendered = f"{surname}, {year}"
            else:
                rendered = surname
            lookup[key] = rendered
    return lookup


def _latex_inline_to_text(text: str, *, bib: dict[str, str]) -> str:
    """Convert common TeX constructs into deterministic source text."""
    text = text.replace("\u00a0", " ").replace("~", " ")
    math_pattern = re.compile(r"\\\((.*?)\\\)|\$\$(.*?)\$\$|\$(.*?)\$|\\\[(.*?)\\\]", flags=re.S)
    placeholders: list[str] = []

    def hold_math(match: re.Match[str]) -> str:
        value = next((group for group in match.groups() if group is not None), "")
        token = f" __MATH{len(placeholders)}__ "
        placeholders.append(normalize_text(value))
        return token

    text = math_pattern.sub(hold_math, text)

    cite_pattern = re.compile(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)?\{([^{}]+)\}")

    def replace_cite(match: re.Match[str]) -> str:
        keys = [part.strip() for part in match.group(1).split(",")]
        rendered = [bib.get(key, key) for key in keys]
        return "(" + "; ".join(rendered) + ")"

    text = cite_pattern.sub(replace_cite, text)

    # Formatting commands: retain argument text, discard command syntax.
    for command in (
        "textbf", "textit", "texttt", "textrm", "textsf", "emph", "underline",
        "textsc", "textnormal", "mbox", "fbox",
    ):
        text = re.sub(rf"\\{command}\s*\{{([^{{}}]*)\}}", r"\1", text)
    text = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:autoref|ref|label)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:textsuperscript|textsubscript|footnote)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:ldots|dots|cdots)", "...", text)
    text = re.sub(r"\\(?:&|%|#|_|\{|\})", lambda m: m.group(0)[1], text)
    text = re.sub(r"\\[a-zA-Z@]+\*?", "", text)
    text = text.replace("{", "").replace("}", "")

    for index, value in enumerate(placeholders):
        text = text.replace(f"__MATH{index}__", value)
    return normalize_text(text)


def _latex_to_qasa_html(source: str, *, paper_id: str, title_hint: str, bib: dict[str, str]) -> str:
    """Convert TeX source to the minimal QASA section/paragraph HTML structure."""
    source = _strip_latex_comments(source)
    start = source.find("\\begin{document}")
    end = source.rfind("\\end{document}")
    if start >= 0:
        source = source[start + len("\\begin{document}") : end if end > start else None]

    for env in (
        "figure", "figure*", "table", "table*", "algorithm", "algorithm*",
        "equation", "equation*", "align", "align*", "alignat", "alignat*",
        "gather", "gather*", "multline", "multline*", "lstlisting", "verbatim",
    ):
        source = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
            "\n",
            source,
            flags=re.S,
        )

    abstract = ""
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, flags=re.S)
    if abstract_match:
        abstract = _latex_inline_to_text(abstract_match.group(1), bib=bib)
        source = source[:abstract_match.start()] + source[abstract_match.end():]

    section_re = re.compile(r"\\section\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}", flags=re.S)
    matches = list(section_re.finditer(source))
    pieces = [
        '<html><head><meta name="qasa-source" content="1"></head><body>',
        f'<h1 class="ltx_title_document">{html_escape(title_hint or paper_id)}</h1>',
    ]
    if abstract:
        pieces.append(f'<div class="ltx_abstract">{html_escape(abstract)}</div>')

    for index, match in enumerate(matches):
        heading = _latex_inline_to_text(match.group(1), bib=bib)
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[body_start:body_end]
        body = re.sub(
            r"\\(?:subsection|subsubsection|paragraph|subparagraph)\*?\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}",
            "\n",
            body,
            flags=re.S,
        )
        body = re.sub(r"\\(?:maketitle|clearpage|newpage|pagebreak|noindent)\b", "\n", body)

        paragraphs = []
        for raw_paragraph in re.split(r"\n\s*\n+", body):
            text = _latex_inline_to_text(raw_paragraph, bib=bib)
            text = re.sub(r"\\item\s*", "", text)
            if text:
                paragraphs.append(text)

        section_id = index + 1
        pieces.append(
            f'<section id="S{section_id}"><h2 class="ltx_title">{html_escape(heading)}</h2>'
        )
        for p_index, paragraph in enumerate(paragraphs):
            pieces.append(
                f'<div class="ltx_para" id="S{section_id}.p{p_index}">'
                f'<p>{html_escape(paragraph)}</p></div>'
            )
        pieces.append("</section>")
    pieces.append("</body></html>")
    return "".join(pieces)


def _source_to_qasa_html(data: bytes, *, paper_id: str, title_hint: str) -> str:
    """Convert an arXiv source archive to QASA-compatible HTML."""
    files = _extract_source_files(data)
    main = _find_main_tex(files)
    if main is None:
        raise ValueError("arXiv source archive does not contain a main TeX document")
    main_name, main_text = main
    expanded = _resolve_tex_inputs(main_text, files=files, current_path=main_name, seen={main_name})
    bib = _load_bib_authors(files)

    try:
        completed = subprocess.run(
            ["pandoc", "-f", "latex", "-t", "html5", "--wrap=none"],
            input=expanded,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            soup = BeautifulSoup(completed.stdout, "html.parser")
            # Pandoc does not always emit the LaTeX abstract in the HTML body.
            # Recover it directly from the expanded source so source-based retrieval
            # remains structurally comparable to ar5iv.
            abstract_match = re.search(
                r"\\begin\{abstract\}(.*?)\\end\{abstract\}", expanded, flags=re.S
            )
            source_abstract = (
                _latex_inline_to_text(abstract_match.group(1), bib=bib)
                if abstract_match
                else ""
            )

            # Pandoc emits citation placeholders when no citeproc bibliography is supplied.
            # Reinsert deterministic author/year text from the arXiv source bibliography.
            for citation in soup.find_all(class_="citation"):
                keys = [key.strip() for key in str(citation.get("data-cites") or "").split()]
                rendered = [bib[key] for key in keys if key in bib]
                if rendered:
                    citation.clear()
                    citation.append("(" + "; ".join(rendered) + ")")

            body = soup.body or soup
            heading_nodes = body.find_all("h1")
            if not heading_nodes:
                # Some Pandoc versions/templates use h2 for top-level sections.
                heading_nodes = body.find_all("h2")
            top_heading_tag = "h1" if body.find("h1") is not None else "h2"

            pieces = [
                '<html><head><meta name="qasa-source" content="pandoc"></head><body>',
                f'<h1 class="ltx_title_document">{html_escape(title_hint or paper_id)}</h1>',
            ]
            if source_abstract:
                pieces.append(f'<div class="ltx_abstract">{html_escape(source_abstract)}</div>')

            sections: list[tuple[str, list[str]]] = []
            current_heading: str | None = None
            current_paragraphs: list[str] = []

            def flush_section() -> None:
                nonlocal current_heading, current_paragraphs
                if current_heading is not None:
                    sections.append((current_heading, current_paragraphs))
                current_heading = None
                current_paragraphs = []

            # Walk the complete block tree in document order. In particular, do not
            # require <p> to be a direct child of the body: Pandoc places list-item
            # paragraphs inside <li>, which otherwise silently disappears here.
            for node in body.find_all(["h1", "h2", "p", "li"]):
                if node.name == top_heading_tag:
                    flush_section()
                    current_heading = normalize_text(node.get_text(" ", strip=True))
                    continue
                if current_heading is None:
                    continue
                # Nested list items containing <p> are captured by their child <p>.
                if node.name == "li" and node.find("p") is not None:
                    continue
                if node.find_parent(["figure", "table"]):
                    continue
                text = normalize_text(node.get_text(" ", strip=True))
                if text:
                    current_paragraphs.append(text)

            flush_section()
            for section_index, (heading, paragraphs) in enumerate(sections, start=1):
                if not heading:
                    continue
                pieces.append(
                    f'<section id="S{section_index}"><h2 class="ltx_title">{html_escape(heading)}</h2>'
                )
                for p_index, text in enumerate(paragraphs):
                    pieces.append(
                        f'<div class="ltx_para" id="S{section_index}.p{p_index}">'
                        f'<p>{html_escape(text)}</p></div>'
                    )
                pieces.append("</section>")
            pieces.append("</body></html>")
            if sections:
                return "".join(pieces)
    except (OSError, subprocess.SubprocessError):
        pass

    return _latex_to_qasa_html(expanded, paper_id=paper_id, title_hint=title_hint, bib=bib)


def _source_evidence_score(data: bytes, evidence_texts: Iterable[str]) -> tuple[int, int, str]:
    """Score an arXiv source archive after deterministic TeX reconstruction."""
    texts = []
    seen: set[str] = set()
    for evidence in evidence_texts:
        normalized = normalize_text(evidence)
        if normalized and normalized not in seen:
            seen.add(normalized)
            texts.append(normalized)
    if not texts:
        return (0, 0, "")
    try:
        html = _source_to_qasa_html(data, paper_id="source", title_hint="Source")
    except Exception:
        return (0, len(texts), "")
    matched, total = _html_evidence_score(html, texts)
    return matched, total, html


def _discover_arxiv_versions(paper_id: str, *, timeout: int = 60) -> list[int]:
    """Return the known arXiv version numbers for a paper, newest first."""
    try:
        response = requests.get(
            f"https://arxiv.org/abs/{paper_id}",
            timeout=timeout,
            verify=True,
            headers={"User-Agent": "sclc-retrieval/1.0 (QASA version audit; research use)"},
        )
        response.raise_for_status()
        versions = {int(value) for value in re.findall(r"\[v(\d+)\]", response.text)}
        return sorted(versions, reverse=True)
    except Exception:
        return []


def _cache_name(paper_id: str, version: int | None) -> str:
    safe = paper_id.replace("/", "_")
    return f"{safe}.html" if version is None else f"{safe}.v{version}.html"


def _source_cache_name(paper_id: str, version: int) -> str:
    safe = paper_id.replace("/", "_")
    return f"{safe}.v{version}.source"


def _source_has_main_tex(data: bytes) -> bool:
    if not data:
        return False
    try:
        return _find_main_tex(_extract_source_files(data)) is not None
    except Exception:
        return False


def _semantic_scholar_open_access_pdf(s2orc_id: str | None, *, timeout: int) -> str | None:
    """Resolve a QASA S2ORC/Corpus id to an open-access PDF URL when available.

    QASA v1.1 contains a small number of papers without an arXiv identifier.
    Their ``s2orc_id`` corresponds to Semantic Scholar's corpus id, which can be
    queried through the Academic Graph API. This lookup is metadata-only; the
    returned PDF is still downloaded and validated locally before use.
    """
    if not s2orc_id:
        return None
    endpoint = f"https://api.semanticscholar.org/graph/v1/paper/CorpusId:{s2orc_id}"
    try:
        response = requests.get(
            endpoint,
            params={"fields": "title,openAccessPdf,externalIds,url"},
            timeout=timeout,
            verify=True,
            headers={"User-Agent": "sclc-retrieval/1.0 (QASA S2ORC fallback; research use)"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    pdf = payload.get("openAccessPdf") or {}
    if isinstance(pdf, dict):
        url = pdf.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _download_response(url: str, *, timeout: int, user_agent: str) -> requests.Response:
    response = requests.get(
        url,
        timeout=timeout,
        verify=True,
        headers={"User-Agent": user_agent},
    )
    response.raise_for_status()
    return response


def fetch_paper_html(
    *,
    paper_id: str,
    url: str | None,
    cache_dir: Path,
    timeout: int = 60,
    evidence_texts: Iterable[str] = (),
    title_hint: str = "",
    s2orc_id: str | None = None,
) -> str:
    """Fetch a QASA paper with version-aware source reconstruction.

    Current ar5iv is preferred. When its text no longer reproduces the QASA
    evidence annotations, historical arXiv versions are checked. ar5iv
    versioned ar5iv URLs are attempted first. If those do not reproduce the
    annotations, historical arXiv source archives are reconstructed into the
    same minimal paragraph/section HTML shape; versioned PDFs remain a final
    fallback when source reconstruction is unavailable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    evidence_texts = tuple(evidence_texts)

    candidates: list[tuple[int | None, str]] = []
    if url:
        candidates.append((None, url))
    canonical = f"https://ar5iv.labs.arxiv.org/html/{paper_id}"
    if all(candidate_url != canonical for _, candidate_url in candidates):
        candidates.append((None, canonical))
    alternate = f"https://ar5iv.org/html/{paper_id}"
    if all(candidate_url != alternate for _, candidate_url in candidates):
        candidates.append((None, alternate))

    fetched: list[tuple[int | None, str, tuple[int, int], str]] = []
    for version, candidate_url in candidates:
        cache_path = cache_dir / _cache_name(paper_id, version)
        html: str | None = None
        if cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            if _looks_like_qasa_full_text(cached):
                html = cached
            else:
                cache_path.unlink()
        if html is None:
            try:
                response = _download_response(
                    candidate_url,
                    timeout=timeout,
                    user_agent="sclc-retrieval/1.0 (QASA preparation; research use)",
                )
                html = response.text
                if not _looks_like_qasa_full_text(html):
                    raise ValueError("Downloaded page is not a parsable QASA/ar5iv body")
                cache_path.write_text(html, encoding="utf-8")
            except Exception:
                continue
        fetched.append((version, candidate_url, _html_evidence_score(html, evidence_texts), html))

    # Even when current ar5iv is completely unavailable, a historical arXiv
    # source archive may still be available. Do not raise before attempting
    # that path: some QASA papers have no usable ar5iv HTML at all.
    versions: list[int] = []
    if not fetched:
        versions = _discover_arxiv_versions(paper_id, timeout=timeout) or [1]
        source_cache_dir = cache_dir / "source"
        source_cache_dir.mkdir(parents=True, exist_ok=True)
        for version in versions:
            source_url = f"https://export.arxiv.org/e-print/{paper_id}v{version}"
            source_path = source_cache_dir / _source_cache_name(paper_id, version)
            data: bytes | None = None
            if source_path.exists():
                cached_bytes = source_path.read_bytes()
                if cached_bytes and _source_has_main_tex(cached_bytes):
                    data = cached_bytes
                elif cached_bytes:
                    source_path.unlink()

            selected_source_url = source_url
            if data is None:
                for candidate_source_url in (
                    source_url,
                    f"https://arxiv.org/e-print/{paper_id}v{version}",
                ):
                    try:
                        response = _download_response(
                            candidate_source_url,
                            timeout=timeout,
                            user_agent="sclc-retrieval/1.0 (QASA source reconstruction; research use)",
                        )
                        candidate_data = response.content or None
                        if candidate_data is not None and _source_has_main_tex(candidate_data):
                            data = candidate_data
                            selected_source_url = candidate_source_url
                            source_path.write_bytes(data)
                            break
                        data = None
                    except Exception:
                        data = None
            if data is None:
                continue

            matched, source_total, source_html = _source_evidence_score(
                data, evidence_texts
            )
            if not source_html:
                continue
            fetched.append((version, selected_source_url, (matched, source_total), source_html))
            if matched == source_total and source_total:
                break

    # If current HTML and historical source reconstruction both failed,
    # continue into the versioned-PDF path below rather than aborting.
    # Some arXiv papers expose usable PDFs even when ar5iv and source
    # conversion are unavailable or incompatible with our parser.
    versions_for_pdf = (_discover_arxiv_versions(paper_id, timeout=timeout) or [1]) if not fetched else []

    if not fetched and not versions_for_pdf:
        raise RuntimeError(
            f"Unable to obtain a usable full-text HTML page for QASA paper {paper_id}."
        )

    if fetched:
        best = max(
            fetched,
            key=lambda item: (
                item[2][0],
                item[2][1],
                1 if item[0] is None else 0,
                item[0] if item[0] is not None else -1,
            ),
        )
    else:
        best = (None, "", (0, len(list(evidence_texts))), "")
    best_version, best_url, best_score, best_html = best
    total = best_score[1]

    if total and best_score[0] < total:
        if not versions:
            versions = _discover_arxiv_versions(paper_id, timeout=timeout) or [1]
        for version in versions:
            if any(existing_version == version for existing_version, _, _, _ in fetched):
                continue

            # First try versioned ar5iv. Some mirrors silently discard the
            # suffix, so the score will reveal whether it was actually useful.
            version_url = f"https://ar5iv.labs.arxiv.org/html/{paper_id}v{version}"
            version_cache = cache_dir / _cache_name(paper_id, version)
            html = None
            if version_cache.exists():
                cached = version_cache.read_text(encoding="utf-8")
                if _looks_like_qasa_full_text(cached):
                    html = cached
                else:
                    version_cache.unlink()
            if html is None:
                try:
                    response = _download_response(
                        version_url,
                        timeout=timeout,
                        user_agent="sclc-retrieval/1.0 (QASA version audit; research use)",
                    )
                    html = response.text
                    if not _looks_like_qasa_full_text(html):
                        html = None
                    else:
                        version_cache.write_text(html, encoding="utf-8")
                except Exception:
                    html = None
            if html is not None:
                score = _html_evidence_score(html, evidence_texts)
                fetched.append((version, version_url, score, html))
                if score[0] == total and total:
                    break

        # If historical HTML still misses evidence, use the source archive itself.
        # This preserves original TeX paragraph boundaries, which can be lost when
        # rendered PDF/HTML extraction is interrupted by figures or tables.
        if fetched:
            best = max(
                fetched,
                key=lambda item: _candidate_selection_key(item, evidence_texts),
            )
            best_version, best_url, best_score, best_html = best
        else:
            total = len({normalize_text(text) for text in evidence_texts if normalize_text(text)})
            best = (None, "", (0, total), "")
            best_version, best_url, best_score, best_html = best
        if best_score[0] < total:
            source_cache_dir = cache_dir / "source"
            source_cache_dir.mkdir(parents=True, exist_ok=True)
            for version in versions:
                source_url = f"https://export.arxiv.org/e-print/{paper_id}v{version}"
                source_path = source_cache_dir / _source_cache_name(paper_id, version)
                data: bytes | None = None
                if source_path.exists():
                    cached_bytes = source_path.read_bytes()
                    if cached_bytes and _source_has_main_tex(cached_bytes):
                        data = cached_bytes
                    elif cached_bytes:
                        source_path.unlink()

                selected_source_url = source_url
                if data is None:
                    for candidate_source_url in (
                        source_url,
                        f"https://arxiv.org/e-print/{paper_id}v{version}",
                    ):
                        try:
                            response = _download_response(
                                candidate_source_url,
                                timeout=timeout,
                                user_agent="sclc-retrieval/1.0 (QASA source reconstruction; research use)",
                            )
                            candidate_data = response.content or None
                            if candidate_data is not None and _source_has_main_tex(candidate_data):
                                data = candidate_data
                                selected_source_url = candidate_source_url
                                source_path.write_bytes(data)
                                break
                            data = None
                        except Exception:
                            data = None
                if data is None:
                    continue

                matched, source_total, source_html = _source_evidence_score(
                    data, evidence_texts
                )
                if not source_html:
                    continue
                source_score = (matched, source_total)
                source_candidate = (version, selected_source_url, source_score, source_html)
                fetched.append(source_candidate)
                if source_score[0] > best_score[0] or (
                    source_score[0] == best_score[0] and source_score[0] > 0
                ):
                    best = source_candidate
                    best_version, best_url, best_score, best_html = best
                if best_score[0] == total and total:
                    break

        # If no historical source/HTML improved the match, fall back to versioned PDFs.
        if fetched:
            best = max(
                fetched,
                key=lambda item: _candidate_selection_key(item, evidence_texts),
            )
            best_version, best_url, best_score, best_html = best
        else:
            best = (None, "", (0, total), "")
            best_version, best_url, best_score, best_html = best
        if best_score[0] < total:
            for version in versions:
                # A versioned ar5iv page may be byte-identical to the current
                # paper, so its presence in ``fetched`` does not mean the
                # corresponding PDF has been checked.
                pdf_url = f"https://arxiv.org/pdf/{paper_id}v{version}"
                pdf_path = cache_dir / f"{paper_id.replace('/', '_')}.v{version}.pdf"
                data: bytes | None = None
                if pdf_path.exists():
                    cached_bytes = pdf_path.read_bytes()
                    if _looks_like_qasa_pdf(cached_bytes):
                        data = cached_bytes
                    else:
                        pdf_path.unlink()
                if data is None:
                    try:
                        response = _download_response(
                            pdf_url,
                            timeout=timeout,
                            user_agent="sclc-retrieval/1.0 (QASA version audit; research use)",
                        )
                        data = response.content
                        if not _looks_like_qasa_pdf(data):
                            data = None
                        else:
                            pdf_path.write_bytes(data)
                    except Exception:
                        data = None
                if data is None:
                    continue

                score = _pdf_evidence_score(data, evidence_texts)
                if score[0] <= best_score[0]:
                    continue
                try:
                    html = _pdf_to_qasa_html(data, paper_id=paper_id, title_hint=title_hint or paper_id)
                    # Re-score using the same parser that will later perform
                    # paragraph alignment.
                    html_score = _html_evidence_score(html, evidence_texts)
                    fetched.append((version, pdf_url, html_score, html))
                    best = max(
                fetched,
                key=lambda item: _candidate_selection_key(item, evidence_texts),
            )
                    best_version, best_url, best_score, best_html = best
                    if best_score[0] == total and total:
                        break
                except Exception:
                    continue

    # When there was no usable HTML/source candidate, run the versioned PDF
    # fallback explicitly. This is needed for papers where ar5iv/source
    # conversion is unavailable but the PDF is downloadable.
    if not fetched and versions_for_pdf:
        for version in versions_for_pdf:
            pdf_url = f"https://arxiv.org/pdf/{paper_id}v{version}"
            pdf_path = cache_dir / f"{paper_id.replace('/', '_')}.v{version}.pdf"
            data: bytes | None = None
            if pdf_path.exists():
                cached_bytes = pdf_path.read_bytes()
                if _looks_like_qasa_pdf(cached_bytes):
                    data = cached_bytes
                else:
                    pdf_path.unlink()
            if data is None:
                try:
                    response = _download_response(
                        pdf_url,
                        timeout=timeout,
                        user_agent="sclc-retrieval/1.0 (QASA PDF fallback; research use)",
                    )
                    data = response.content
                    if not _looks_like_qasa_pdf(data):
                        data = None
                    else:
                        pdf_path.write_bytes(data)
                except Exception:
                    data = None
            if data is None:
                continue
            try:
                html = _pdf_to_qasa_html(data, paper_id=paper_id, title_hint=title_hint or paper_id)
                html_score = _html_evidence_score(html, evidence_texts)
                fetched.append((version, pdf_url, html_score, html))
                best = max(
                fetched,
                key=lambda item: _candidate_selection_key(item, evidence_texts),
            )
                best_version, best_url, best_score, best_html = best
                total = best_score[1]
                if total and best_score[0] == total:
                    break
            except Exception:
                continue

    # QASA v1.1 includes a small non-arXiv subset. Prefer a locally cached
    # S2ORC PDF before any network metadata lookup so that manually cached
    # public PDFs are sufficient and rate limits cannot block preparation.
    cached_s2orc_used = False
    if not fetched or best_score[0] < total:
        safe_paper_id = paper_id.replace("/", "_")
        pdf_path = cache_dir / f"{safe_paper_id}.s2orc.pdf"
        pdf_data: bytes | None = None
        if pdf_path.exists():
            cached_bytes = pdf_path.read_bytes()
            if _looks_like_qasa_pdf(cached_bytes):
                pdf_data = cached_bytes
            else:
                try:
                    pdf_path.unlink()
                except OSError:
                    pass
        if pdf_data is not None:
            try:
                pdf_html = _pdf_to_qasa_html(
                    pdf_data,
                    paper_id=paper_id,
                    title_hint=title_hint or paper_id,
                )
                pdf_score = _html_evidence_score(pdf_html, evidence_texts)
                fetched.append((None, str(pdf_path), pdf_score, pdf_html))
                best = max(
                    fetched,
                    key=lambda item: _candidate_selection_key(item, evidence_texts),
                )
                best_version, best_url, best_score, best_html = best
                total = best_score[1]
                cached_s2orc_used = True
            except Exception:
                pass

    # Only consult Semantic Scholar when there is no usable local S2ORC PDF.
    # A locally supplied PDF is an explicit reproducibility input and avoids
    # making the preparation dependent on API availability or rate limits.
    if (not cached_s2orc_used) and (not fetched or best_score[0] < total) and s2orc_id:
        oa_pdf_url = _semantic_scholar_open_access_pdf(s2orc_id, timeout=timeout)
        if oa_pdf_url:
            safe_paper_id = paper_id.replace("/", "_")
            pdf_path = cache_dir / f"{safe_paper_id}.s2orc.pdf"
            pdf_data: bytes | None = None
            try:
                response = _download_response(
                    oa_pdf_url,
                    timeout=timeout,
                    user_agent="sclc-retrieval/1.0 (QASA S2ORC PDF fallback; research use)",
                )
                if _looks_like_qasa_pdf(response.content):
                    pdf_data = response.content
                    pdf_path.write_bytes(pdf_data)
            except Exception:
                pdf_data = None
            if pdf_data is not None:
                try:
                    pdf_html = _pdf_to_qasa_html(
                        pdf_data,
                        paper_id=paper_id,
                        title_hint=title_hint or paper_id,
                    )
                    pdf_score = _html_evidence_score(pdf_html, evidence_texts)
                    fetched.append((None, oa_pdf_url, pdf_score, pdf_html))
                    best = max(
                        fetched,
                        key=lambda item: _candidate_selection_key(item, evidence_texts),
                    )
                    best_version, best_url, best_score, best_html = best
                except Exception:
                    pass

    if not fetched:
        raise RuntimeError(
            f"Unable to obtain a usable full-text representation for QASA paper {paper_id}."
        )

    if total and best_score[0] == 0:
        print(
            f"[QASA] Warning: no evidence paragraphs matched for {paper_id}; "
            f"best source {best_url} matched 0/{total}."
        )
    elif total and best_score[0] < total:
        print(
            f"[QASA] Partial evidence alignment for {paper_id}: "
            f"selected {best_url} with {best_score[0]}/{total} exact evidence paragraphs."
        )
    elif total:
        print(
            f"[QASA] Selected {best_url} for {paper_id}: "
            f"{best_score[0]}/{total} evidence paragraphs aligned."
        )

    return best_html

def iter_qasa_documents(
    config: AppConfig,
    *,
    json_path: Path | None = None,
    split: str = "test",
    limit: int | None = None,
) -> list[DocumentRecord]:
    source = json_path or Path(config.dataset.raw_json_path or "data/qasa/raw/testset_answerable_1554_v1.1.json")
    if not source.exists():
        raise FileNotFoundError(
            f"QASA JSON not found at {source}. Download the official v1.1 answerable file "
            f"from {QASA_DATASET_URL} first."
        )

    items = load_qasa_items(source)
    grouped: dict[str, list[QASAItem]] = defaultdict(list)
    paper_urls: dict[str, str | None] = {}
    paper_s2orc_ids: dict[str, str | None] = {}
    for item in items:
        grouped[item.paper_id].append(item)
        paper_urls.setdefault(item.paper_id, item.paper_url)
        paper_s2orc_ids.setdefault(item.paper_id, item.s2orc_id)

    paper_cache = Path(
        config.dataset.qasa_paper_cache_dir
        or config.paths.processed_dir.parent / "papers"
    )
    documents: list[DocumentRecord] = []
    iterable = list(grouped.items())
    if limit is not None:
        iterable = iterable[:limit]

    for paper_id, query_items in tqdm(iterable, desc="Preparing QASA papers"):
        evidence_texts = [
            text
            for item in query_items
            for text in item.evidence_texts
        ]
        html = fetch_paper_html(
            paper_id=paper_id,
            url=paper_urls.get(paper_id),
            cache_dir=paper_cache,
            timeout=int(config.dataset.qasa_request_timeout_seconds),
            evidence_texts=evidence_texts,
            title_hint=next((item.title for item in query_items if item.title), ""),
            s2orc_id=paper_s2orc_ids.get(paper_id),
        )
        documents.append(
            _parse_qasa_html(
                paper_id=paper_id,
                html=html,
                split=split,
                title_hint=next((item.title for item in query_items if item.title), ""),
                config=config,
                query_items=query_items,
            )
        )
    return documents
