from local_rag_notebook.app import build_contexts
from local_rag_notebook.index.schema import Chunk


def _mk_chunk(chunk_id: str, text: str, page_no: int = 1) -> Chunk:
    return Chunk(
        doc_id="doc.pdf",
        chunk_id=chunk_id,
        level="section",
        heading_path="doc.pdf > Page 1",
        page_no=page_no,
        text=text,
        meta={"file_path": "doc.pdf"},
    )


def test_build_contexts_does_not_truncate_long_text():
    long_text = "A" * 5000  # far past the old hardcoded 1000-char cutoff
    chunk = _mk_chunk("c1", long_text)

    contexts = build_contexts([chunk], score_map={}, top_k=8)

    assert len(contexts[0]["text"]) == 5000
    assert contexts[0]["text"] == long_text


def test_build_contexts_uses_real_score_when_available():
    c1 = _mk_chunk("c1", "alpha")
    c2 = _mk_chunk("c2", "beta")

    contexts = build_contexts([c1, c2], score_map={"c1": 0.83, "c2": 0.21}, top_k=8)

    scores = {c["id"]: c["score"] for c in contexts}
    assert scores["c1"] == 0.83
    assert scores["c2"] == 0.21
    # Real scores must vary across chunks, not collapse to one constant.
    assert scores["c1"] != scores["c2"]


def test_build_contexts_score_none_when_not_a_dense_hit():
    # e.g. a chunk pulled in only via neighbor expansion
    c1 = _mk_chunk("c1", "alpha")

    contexts = build_contexts([c1], score_map={}, top_k=8)

    assert contexts[0]["score"] is None
