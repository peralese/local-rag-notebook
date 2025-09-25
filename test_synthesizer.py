from __future__ import annotations

import pytest

from synthesizer import validate_citations, decide_abstain


def test_validate_citations_non_strict_ok():
    parsed = {
        "answer_markdown": "Hello [C1]. World [C2]!",
        "citations": [{"id": "C1"}, {"id": "C2"}],
    }
    ok, reason = validate_citations(parsed, strict=False)
    assert ok, reason


def test_validate_citations_missing_tag_fails():
    parsed = {
        "answer_markdown": "Hello [C1]. World [C2]!",
        "citations": [{"id": "C1"}],
    }
    ok, reason = validate_citations(parsed, strict=False)
    assert not ok
    assert "not in citations array" in reason


def test_validate_citations_strict_requires_tag_each_sentence():
    parsed = {
        "answer_markdown": "Sentence with tag [C1]. Sentence without tag.",
        "citations": [{"id": "C1"}],
    }
    ok, reason = validate_citations(parsed, strict=True)
    assert not ok
    assert "sentence lacks citation tag" in reason


@pytest.mark.parametrize(
    "cov,avg,thresh,expect_abstain",
    [
        (1.0, 1.0, 0.70, False),
        (0.2, 0.2, 0.70, True),
        (0.6, 0.8, 0.70, False),  # blended = 0.7
        (0.0, 0.7, 0.71, True),   # blended = 0.35
    ],
)
def test_decide_abstain_blended(cov, avg, thresh, expect_abstain):
    parsed = {"support_coverage": cov, "abstain": False}
    abstain, why = decide_abstain(parsed, avg_sim=avg, threshold=thresh)
    assert abstain is expect_abstain
