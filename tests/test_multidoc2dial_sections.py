from sclc.data.multidoc2dial import _normalize_disjoint_top_sections


def test_multidoc2dial_top_sections_are_disjoint():
    sections = [
        {"start": 0, "end": 100, "title": "A"},
        {"start": 60, "end": 160, "title": "B"},
        {"start": 140, "end": 220, "title": "C"},
    ]

    result = _normalize_disjoint_top_sections(sections, 220)

    assert [(x["start"], x["end"]) for x in result] == [
        (0, 60),
        (60, 140),
        (140, 220),
    ]

    for left, right in zip(result, result[1:]):
        assert left["end"] <= right["start"]
