from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from sclc.data.schema import PreparedQueryRecord

SINGLE_PARAGRAPH = "single_paragraph"
MULTI_PARAGRAPH_SAME_SECTION = "multi_paragraph_same_section"
MULTI_PARAGRAPH_CROSS_SECTION = "multi_paragraph_cross_section"

EVIDENCE_STRUCTURES: tuple[str, ...] = (
    SINGLE_PARAGRAPH,
    MULTI_PARAGRAPH_SAME_SECTION,
    MULTI_PARAGRAPH_CROSS_SECTION,
)

_STRUCTURE_ORDER = {label: index for index, label in enumerate(EVIDENCE_STRUCTURES)}


def _structure_for_set(
    paragraph_ids: Sequence[str],
    paragraph_to_parent_section: Mapping[str, str],
) -> tuple[str, int, int]:
    unique_ids = sorted(set(paragraph_ids))
    if not unique_ids:
        raise ValueError("An acceptable evidence set cannot be empty")

    missing = [
        paragraph_id
        for paragraph_id in unique_ids
        if paragraph_id not in paragraph_to_parent_section
    ]
    if missing:
        raise ValueError(
            "Evidence paragraphs could not be mapped to top-level sections: "
            f"{missing[:10]}"
        )

    parent_sections = {
        paragraph_to_parent_section[paragraph_id] for paragraph_id in unique_ids
    }
    paragraph_count = len(unique_ids)
    section_count = len(parent_sections)

    if paragraph_count == 1:
        structure = SINGLE_PARAGRAPH
    elif section_count == 1:
        structure = MULTI_PARAGRAPH_SAME_SECTION
    else:
        structure = MULTI_PARAGRAPH_CROSS_SECTION
    return structure, paragraph_count, section_count


def classify_evidence_structure(
    query: PreparedQueryRecord,
    paragraph_to_parent_section: Mapping[str, str],
) -> dict[str, Any]:
    """Classify a query by the least-distributed minimal acceptable support set.

    QASPER may provide multiple alternative acceptable evidence sets. Complete-
    support evaluation succeeds when any one set is fully recovered, so the
    primary structure is based on the smallest complete acceptable set. If tied
    minimal sets have different structures, the least-distributed structure is
    selected conservatively and the ambiguity is retained explicitly.
    """

    if not query.evidence_sets:
        raise ValueError(f"Query {query.query_id} has no acceptable evidence sets")

    set_rows: list[dict[str, Any]] = []
    for evidence_set in query.evidence_sets:
        structure, paragraph_count, section_count = _structure_for_set(
            evidence_set.paragraph_ids,
            paragraph_to_parent_section,
        )
        set_rows.append(
            {
                "evidence_set_id": evidence_set.evidence_set_id,
                "paragraph_count": paragraph_count,
                "top_level_section_count": section_count,
                "evidence_structure": structure,
            }
        )

    minimum_paragraph_count = min(row["paragraph_count"] for row in set_rows)
    minimal_rows = [
        row for row in set_rows if row["paragraph_count"] == minimum_paragraph_count
    ]
    minimal_structures = sorted(
        {row["evidence_structure"] for row in minimal_rows},
        key=_STRUCTURE_ORDER.__getitem__,
    )
    primary_structure = minimal_structures[0]
    primary_candidates = [
        row
        for row in minimal_rows
        if row["evidence_structure"] == primary_structure
    ]
    primary_evidence_set_id = sorted(
        str(row["evidence_set_id"]) for row in primary_candidates
    )[0]

    union_structure, union_paragraph_count, union_section_count = _structure_for_set(
        query.evidence_union_paragraph_ids,
        paragraph_to_parent_section,
    )
    all_structures = sorted(
        {row["evidence_structure"] for row in set_rows},
        key=_STRUCTURE_ORDER.__getitem__,
    )

    return {
        "evidence_structure": primary_structure,
        "primary_evidence_set_id": primary_evidence_set_id,
        "minimum_evidence_paragraph_count": minimum_paragraph_count,
        "minimal_evidence_set_count": len(minimal_rows),
        "minimum_top_level_section_count_among_minimal_sets": min(
            row["top_level_section_count"] for row in minimal_rows
        ),
        "maximum_top_level_section_count_among_minimal_sets": max(
            row["top_level_section_count"] for row in minimal_rows
        ),
        "minimal_structure_variants": "|".join(minimal_structures),
        "minimal_structure_ambiguous": len(minimal_structures) > 1,
        "cross_section_required_among_minimal_sets": all(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in minimal_rows
        ),
        "cross_section_required": all(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in set_rows
        ),
        "cross_section_possible": any(
            row["evidence_structure"] == MULTI_PARAGRAPH_CROSS_SECTION
            for row in set_rows
        ),
        "same_section_multi_paragraph_possible": any(
            row["evidence_structure"] == MULTI_PARAGRAPH_SAME_SECTION
            for row in set_rows
        ),
        "single_paragraph_possible": any(
            row["evidence_structure"] == SINGLE_PARAGRAPH for row in set_rows
        ),
        "acceptable_set_structures": "|".join(all_structures),
        "evidence_set_count": len(set_rows),
        "evidence_union_structure": union_structure,
        "evidence_union_paragraph_count": union_paragraph_count,
        "evidence_union_top_level_section_count": union_section_count,
        "evidence_set_details_json": json.dumps(set_rows, sort_keys=True),
    }



__all__ = [
    "EVIDENCE_STRUCTURES",
    "MULTI_PARAGRAPH_CROSS_SECTION",
    "MULTI_PARAGRAPH_SAME_SECTION",
    "SINGLE_PARAGRAPH",
    "classify_evidence_structure",
]
