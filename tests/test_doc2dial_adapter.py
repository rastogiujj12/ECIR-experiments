from sclc.data.doc2dial import _hierarchy_for_span, _make_query_records, _normalize_disjoint_top_sections


def test_doc2dial_parent_titles_string_is_parsed():
    item = {
        "parent_titles": "[{'id_sp': '1', 'text': 'Top', 'level': 'h2'}]",
        "title": "Nested",
    }
    assert _hierarchy_for_span(item) == ["Top", "Nested"]


def test_doc2dial_source_sections_are_disjoint_after_normalization():
    sections = [
        {"title": "A", "start": 0, "end": 60},
        {"title": "B", "start": 50, "end": 80},
    ]
    result = _normalize_disjoint_top_sections(sections, 100)
    assert [(x["start"], x["end"]) for x in result] == [(0, 50), (50, 100)]


def test_doc2dial_queries_use_agent_grounding_for_user_turn():
    dialogues = [
        {
            "dial_id": "d1",
            "turns": [
                {
                    "role": "user",
                    "turn_id": 1,
                    "utterance": "Can I do this online?",
                    "references": [{"sp_id": "u1", "label": "solution"}],
                },
                {
                    "role": "agent",
                    "turn_id": 2,
                    "utterance": "Yes.",
                    "references": [{"sp_id": "a1", "label": "solution"}],
                },
            ],
        }
    ]
    spans = {
        "u1": {"text_sp": "user reference"},
        "a1": {"text_sp": "agent evidence"},
    }
    queries = _make_query_records(
        domain="dmv", doc_id="doc", dialogues=dialogues, spans=spans
    )
    assert len(queries) == 1
    assert queries[0].question == "Can I do this online?"
    assert [e.text for e in queries[0].answers[0].evidence] == ["agent evidence"]
