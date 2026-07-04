from local_rag_notebook.index.schema import Chunk
from local_rag_notebook.retrieve.rerank import Reranker


def _mk(chunk_id: str) -> Chunk:
    return Chunk(
        doc_id="doc",
        chunk_id=chunk_id,
        level="section",
        heading_path="doc > Page 1",
        page_no=1,
        text=chunk_id,
        meta={},
    )


def _reranker_with_scores(scores) -> Reranker:
    # Bypass __init__ (which would try to download a real CrossEncoder model)
    # and inject a fake model whose predict() returns fixed scores in
    # input-candidate order — same shape a real cross-encoder returns.
    rr = Reranker.__new__(Reranker)
    rr.enabled = True
    rr._ensure_model = lambda: None
    rr._model = type("FakeModel", (), {"predict": staticmethod(lambda pairs: list(scores))})()
    return rr


def test_rerank_inverts_input_order_and_score_stays_with_its_chunk():
    # Input order A, B, C; scores force a full reversal: A=lowest, C=highest.
    chunks = [_mk("A"), _mk("B"), _mk("C")]
    rr = _reranker_with_scores([0.1, 0.5, 0.9])
    true_scores = {"A": 0.1, "B": 0.5, "C": 0.9}

    reranked = rr.rerank("q", chunks, top_k=3)

    # Order must actually have been inverted (this is the whole point of
    # reranking) so this test would be meaningless if it hadn't reordered.
    assert [rc.chunk.chunk_id for rc in reranked] == ["C", "B", "A"]

    # Each chunk must carry its OWN true score, not the score of whichever
    # candidate originally sat at that position pre-sort.
    for rc in reranked:
        assert rc.score == true_scores[rc.chunk.chunk_id]


def test_min_score_cutoff_keeps_the_right_chunks_after_reorder():
    chunks = [_mk("A"), _mk("B"), _mk("C")]
    rr = _reranker_with_scores([0.1, 0.5, 0.9])

    reranked = rr.rerank("q", chunks, top_k=3)
    kept = {rc.chunk.chunk_id for rc in reranked if rc.score is not None and rc.score >= 0.5}

    # Must keep the actually-high-scoring chunks (B, C), not whichever
    # chunks happened to land in the same list position as a high score
    # under the old (unfixed) positional zip.
    assert kept == {"B", "C"}


def test_candidates_beyond_top_k_are_preserved_with_no_score():
    chunks = [_mk(c) for c in ["A", "B", "C", "D", "E"]]
    rr = _reranker_with_scores([0.1, 0.5, 0.9])  # only 3 candidates get scored

    reranked = rr.rerank("q", chunks, top_k=3)

    # All 5 input chunks must still be present (the old zip-based pairing
    # silently truncated to the shorter list, dropping D and E).
    assert {rc.chunk.chunk_id for rc in reranked} == {"A", "B", "C", "D", "E"}
    tail = {rc.chunk.chunk_id: rc.score for rc in reranked if rc.chunk.chunk_id in ("D", "E")}
    assert tail == {"D": None, "E": None}


def test_disabled_reranker_returns_chunks_unscored():
    chunks = [_mk("A"), _mk("B")]
    rr = Reranker.__new__(Reranker)
    rr.enabled = False

    reranked = rr.rerank("q", chunks, top_k=2)

    assert [rc.chunk.chunk_id for rc in reranked] == ["A", "B"]
    assert all(rc.score is None for rc in reranked)
