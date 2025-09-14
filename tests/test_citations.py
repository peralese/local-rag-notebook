import pytest
import synthesizer as sz


def _parsed(answer_markdown: str, citations):
    return {
        "answer_markdown": answer_markdown,
        "citations": citations,
        "support_coverage": 0.9,
        "abstain": False,
        "why": "",
    }


def test_validate_citations_missing_tag_in_array():
    parsed = _parsed("Fact. [C1]", citations=[{"id": "C2"}])
    ok, reason = sz.validate_citations(parsed, strict=False)
    assert not ok
    assert "not in citations array" in reason


def test_validate_citations_strict_sentence_without_tag():
    parsed = _parsed("First sentence without cite. Second has one [C1].", citations=[{"id": "C1"}])
    ok, reason = sz.validate_citations(parsed, strict=True)
    assert not ok
    assert "strict mode" in reason
    assert "sentence lacks citation tag" in reason


def test_validate_citations_ok_when_all_sentences_tagged():
    parsed = _parsed("A [C1]. B [C1].", citations=[{"id": "C1"}])
    ok, reason = sz.validate_citations(parsed, strict=True)
    assert ok
    assert reason == ""


def test_decide_abstain_blended_below_threshold():
    # Low avg_sim + low coverage → abstain
    parsed = _parsed("A [C1].", citations=[{"id": "C1"}])
    parsed["support_coverage"] = 0.2
    do, why = sz.decide_abstain(parsed, avg_sim=0.2, threshold=0.7)
    assert do
    assert "insufficient support" in why
