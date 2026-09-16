# Query-Type Coding Guide

Code every retained QASPER question **after the final sampled query set has been fixed and before running retrieval or inspecting rankings**. Use only the question and its annotated evidence locations in `query_type_coding.csv`. Do not use retrieval scores or retrieved units while coding.

Allowed labels:

## `factual`

Use when the answer is a specific fact, value, entity, definition, dataset detail, model setting, or directly stated finding that can normally be supported by one local evidence passage.

Typical signs:

- asks *what*, *which*, *who*, *how many*, or for a named value;
- annotated support is local and does not require combining distinct claims;
- the section containing the answer is not itself the main point of the question.

## `section_specific`

Use when the question explicitly or implicitly targets information serving a recognisable section role, such as methodology, experimental setup, results, limitations, or conclusions, and distinguishing that role matters to relevance.

Typical signs:

- asks how the study was conducted, evaluated, or concluded;
- a similar term may appear elsewhere, but the answer must come from the targeted section role;
- the main retrieval difficulty is selecting the correct document section rather than combining evidence.

## `multi_hop`

Use when a complete answer requires combining two or more distinct evidence statements, paragraphs, or reasoning steps. The evidence may be in one section or several sections.

Typical signs:

- asks for a comparison, relationship, cause, or explanation that depends on multiple facts;
- no single annotated paragraph provides the full answer;
- the evidence set contains multiple complementary paragraphs.

Do not use `multi_hop` merely because an annotator supplied duplicate or alternative evidence passages. The passages must contribute different parts of the answer.

## `synthesis`

Use when the question asks for a broad summary, overall contribution, central conclusion, set of findings, or integrated account of a substantial part of the paper. The answer is wider than a local fact and is best supported by evidence distributed across the document or a large section.

Typical signs:

- asks for the paper's main findings, contributions, conclusions, or overall approach;
- requires bringing together several related claims into a coherent answer;
- breadth and coverage matter more than a short chain of reasoning.

## `uncertain`

Use only when the question cannot be assigned confidently after applying the rules above, or when the annotation does not provide enough information to distinguish two categories. Record the competing labels and reason in `coding_notes`.

`uncertain` questions remain in overall retrieval results but are excluded from query-category comparisons.

## Tie-breaking order

When more than one label seems plausible, apply these rules:

1. Use `synthesis` for broad paper-level or section-level integration.
2. Otherwise use `multi_hop` when distinct evidence statements must be combined.
3. Otherwise use `section_specific` when the section role is essential to interpreting relevance.
4. Otherwise use `factual`.
5. Use `uncertain` when the distinction still cannot be made reliably.

## Coding procedure

1. Work from `data/retrieval_units/query_type_coding.csv`.
2. Read the question, evidence counts, and evidence section headings.
3. Consult the annotated evidence text in the prepared document only when the worksheet is insufficient.
4. Assign exactly one allowed label to every query.
5. Add a short note for ambiguous decisions and every `uncertain` label.
6. Save at least the columns `query_id,query_type` as `data/retrieval_units/query_types.csv`.
7. Check for blanks, duplicate query IDs, and invalid spellings before retrieval.

The final dissertation used single-researcher manual coding with this predefined guide. No independent second coder or inter-rater reliability estimate was used. Query-type findings are therefore treated as supporting analyses, particularly for the smaller multi-hop and synthesis groups.
