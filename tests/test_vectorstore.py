from web.vectorstore import VectorStore

def test_upsert_and_query(tmp_path):
    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    vs.reset()
    vs.add(ids=["a1"], documents=["hello world"], metadatas=[{"path":"x"}])
    vs.add(ids=["a1"], documents=["hello world"], metadatas=[{"path":"x"}])
    assert vs.count() == 1
    out = vs.query("hello", k=1)
    assert out["documents"][0]
