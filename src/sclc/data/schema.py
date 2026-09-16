from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CharacterSpan(BaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def validate_span(self) -> CharacterSpan:
        if self.start < 0:
            raise ValueError("Span start cannot be negative")
        if self.end < self.start:
            raise ValueError("Span end cannot be before span start")
        return self


class ParagraphRecord(BaseModel):
    paragraph_id: str
    section_id: str
    source_section_index: int
    source_paragraph_index: int
    text: str
    span: CharacterSpan


class SectionRecord(BaseModel):
    section_id: str
    source_section_index: int
    heading: str
    top_level_heading: str
    hierarchy: list[str]
    span: CharacterSpan
    paragraph_ids: list[str] = Field(default_factory=list)
    is_abstract: bool = False


class EvidenceItem(BaseModel):
    text: str
    is_float: bool
    matched_paragraph_ids: list[str] = Field(default_factory=list)
    alignment_status: str


class AnswerAnnotation(BaseModel):
    annotation_id: str | None = None
    unanswerable: bool = False
    extractive_spans: list[str] = Field(default_factory=list)
    yes_no: bool | None = None
    free_form_answer: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    highlighted_evidence: list[str] = Field(default_factory=list)


class QueryRecord(BaseModel):
    query_id: str
    question: str
    answers: list[AnswerAnnotation] = Field(default_factory=list)

    @property
    def has_usable_textual_evidence(self) -> bool:
        return any(
            item.matched_paragraph_ids
            for answer in self.answers
            if not answer.unanswerable
            for item in answer.evidence
            if not item.is_float
        )


class DocumentRecord(BaseModel):
    document_id: str
    split: str
    title: str
    abstract: str
    text: str
    sections: list[SectionRecord]
    paragraphs: list[ParagraphRecord]
    queries: list[QueryRecord]

    @property
    def usable_question_count(self) -> int:
        return sum(query.has_usable_textual_evidence for query in self.queries)


class TopLevelSectionRecord(BaseModel):
    parent_section_id: str
    document_id: str
    analysis_set: str
    heading: str
    span: CharacterSpan
    source_section_ids: list[str] = Field(default_factory=list)
    text: str


class RetrievalUnitRecord(BaseModel):
    retrieval_unit_id: str
    document_id: str
    analysis_set: str
    segmentation_plan: str
    unit_index: int
    span: CharacterSpan
    text: str
    token_count: int
    scope_token_start: int
    scope_token_end: int
    parent_section_id: str | None = None
    parent_section_heading: str | None = None
    source_section_ids: list[str] = Field(default_factory=list)
    overlapping_section_ids: list[str] = Field(default_factory=list)
    overlapping_parent_section_ids: list[str] = Field(default_factory=list)
    overlapping_paragraph_ids: list[str] = Field(default_factory=list)
    fully_contained_paragraph_ids: list[str] = Field(default_factory=list)


class EvidenceSetRecord(BaseModel):
    evidence_set_id: str
    paragraph_ids: list[str] = Field(default_factory=list)


class PreparedQueryRecord(BaseModel):
    query_id: str
    document_id: str
    split: str
    analysis_set: str
    question: str
    query_type: str | None = None
    evidence_union_paragraph_ids: list[str] = Field(default_factory=list)
    evidence_sets: list[EvidenceSetRecord] = Field(default_factory=list)


class RelevanceJudgementRecord(BaseModel):
    query_id: str
    document_id: str
    segmentation_plan: str
    retrieval_unit_id: str
    relevance: int = 1
    matched_evidence_paragraph_ids: list[str] = Field(default_factory=list)


class RankingRecord(BaseModel):
    query_id: str
    document_id: str
    analysis_set: str
    condition: str
    model_key: str | None = None
    segmentation_plan: str
    retrieval_unit_id: str
    rank: int
    score: float
    parent_section_id: str | None = None
    character_start: int
    character_end: int
    overlapping_paragraph_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranking(self) -> RankingRecord:
        if self.rank <= 0:
            raise ValueError("Ranking rank must be positive")
        if self.character_end < self.character_start:
            raise ValueError("Ranking character span is invalid")
        return self
