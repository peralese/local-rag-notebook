import synthesizer as sz


def _chunks():
    return [{"text": "Some retrieved content about the topic.", "title": "doc.pdf", "path": "doc.pdf"}]


def test_no_context_reason_when_nothing_retrieved():
    result = sz.synthesize_answer("q", retrieved=[], avg_sim=0.7)
    assert result["abstain"] is True
    assert result["abstain_reason"] == "no_context"


def test_llm_unreachable_reason_is_distinct_from_insufficient_support(monkeypatch):
    def _boom(*args, **kwargs):
        raise ConnectionError("Cannot reach Ollama")

    monkeypatch.setattr(sz, "_call_local_llm", _boom)

    result = sz.synthesize_answer("q", retrieved=_chunks(), avg_sim=0.9)

    assert result["abstain"] is True
    assert result["abstain_reason"] == "llm_unreachable"
    assert "ConnectionError" in result["why"]


def test_insufficient_support_reason_when_blended_score_too_low(monkeypatch):
    def _low_support(*args, **kwargs):
        return {
            "answer_markdown": "A weak claim [C1].",
            "citations": [{"id": "C1", "title": "doc.pdf", "uri_or_path": "doc.pdf"}],
            "support_coverage": 0.1,
            "abstain": False,
            "why": "",
        }

    monkeypatch.setattr(sz, "_call_local_llm", _low_support)

    result = sz.synthesize_answer("q", retrieved=_chunks(), avg_sim=0.1, abstain_threshold=0.7)

    assert result["abstain"] is True
    assert result["abstain_reason"] == "insufficient_support"
    assert result["abstain_reason"] != "llm_unreachable"


def test_citation_invalid_reason_when_validation_fails(monkeypatch):
    def _bad_citation(*args, **kwargs):
        return {
            "answer_markdown": "A claim [C9].",  # C9 not in citations array
            "citations": [{"id": "C1", "title": "doc.pdf", "uri_or_path": "doc.pdf"}],
            "support_coverage": 0.95,
            "abstain": False,
            "why": "",
        }

    monkeypatch.setattr(sz, "_call_local_llm", _bad_citation)

    result = sz.synthesize_answer("q", retrieved=_chunks(), avg_sim=0.9)

    assert result["abstain"] is True
    assert result["abstain_reason"] == "citation_invalid"
