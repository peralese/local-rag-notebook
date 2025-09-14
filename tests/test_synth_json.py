import json
import importlib

# Support both layouts:
#  - synthesizer.py at repo root
#  - local_rag_notebook/synthesizer.py inside the package
try:
    sz = importlib.import_module("synthesizer")
except ModuleNotFoundError:
    sz = importlib.import_module("local_rag_notebook.synthesizer")


def test_json_loads_strict_plain_json():
    data = {
        "answer_markdown": "Hi [C1].",
        "citations": [{"id": "C1"}],
        "support_coverage": 1.0,
        "abstain": False,
        "why": "",
    }
    s = json.dumps(data)
    parsed = sz._json_loads_strict(s)
    assert parsed["answer_markdown"].startswith("Hi")
    assert parsed["citations"][0]["id"] == "C1"


def test_json_loads_strict_with_code_fence_and_noise():
    data = {
        "answer_markdown": "Hello [C2].",
        "citations": [{"id": "C2"}],
        "support_coverage": 0.9,
        "abstain": False,
        "why": "",
    }
    s = "Some preface…\n```json\n" + json.dumps(data) + "\n```\ntrailing text"
    parsed = sz._json_loads_strict(s)
    assert parsed["citations"][0]["id"] == "C2"


def test_extract_first_json_object_with_leading_bom_and_suffix():
    data = {
        "answer_markdown": "Test [C1].",
        "citations": [{"id": "C1"}],
        "support_coverage": 0.8,
        "abstain": False,
        "why": "",
    }
    s = "\ufeffnote\n   " + json.dumps(data) + "   garbage after"
    out = sz._extract_first_json_object(s)
    assert out["answer_markdown"].startswith("Test")


def test_extract_first_json_object_unbalanced_raises():
    import pytest

    with pytest.raises(ValueError):
        sz._extract_first_json_object('prefix {"a": 1  suffix without close')

