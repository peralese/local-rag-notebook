import importlib

try:
    sz = importlib.import_module("synthesizer")
except ModuleNotFoundError:
    sz = importlib.import_module("local_rag_notebook.synthesizer")


def _mk(text, title, path):
    return {"text": text, "title": title, "path": path}


def test_pack_context_enforces_per_source_quota():
    chunks = [
        _mk(f"text {i}", "DocA", "a.pdf") for i in range(10)
    ] + [
        _mk("other doc text", "DocB", "b.pdf")
    ]
    packed = sz.pack_context(chunks, max_chars=10_000, per_source_quota=2)
    # Should keep at most 2 from a.pdf, plus 1 from b.pdf (order preserved)
    assert sum(1 for c in packed if c["uri_or_path"].endswith("a.pdf")) == 2
    assert sum(1 for c in packed if c["uri_or_path"].endswith("b.pdf")) == 1


def test_pack_context_filters_near_duplicates():
    base = "This is the first chunk with important terms alpha beta gamma."
    near = "This is the first chunk with important terms alpha beta gamma!"  # near-identical
    far = "Completely different content about delta epsilon zeta."

    chunks = [
        _mk(base, "DocA", "a.pdf"),
        _mk(near, "DocA", "a.pdf"),
        _mk(far, "DocA", "a.pdf"),
    ]
    packed = sz.pack_context(chunks, max_chars=10_000, per_source_quota=3, near_dup_jaccard=0.85)
    texts = [c["text"] for c in packed]
    # The near-duplicate should be filtered; 'far' should remain
    assert base in texts
    assert far in texts
    assert near not in texts


def test_pack_context_respects_char_budget():
    long = "x" * 1_000
    chunks = [
        _mk(long, "DocA", "a.pdf"),
        _mk(long, "DocB", "b.pdf"),
        _mk(long, "DocC", "c.pdf"),
    ]
    packed = sz.pack_context(chunks, max_chars=1500, per_source_quota=5)
    # First fits, second would exceed 1500 → only first kept
    assert len(packed) == 1
