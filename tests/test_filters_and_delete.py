from web.vectorstore import VectorStore
from web.ingest import ingest_file
import os, time

def test_filters_and_delete(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    f1 = data / "a.txt"
    f2 = data / "b.txt"
    f1.write_text("alpha apples banana", encoding="utf-8")
    f2.write_text("alpha oranges pear", encoding="utf-8")

    vs = VectorStore(persist_dir=str(tmp_path / "chroma"))
    vs.reset()
    ingest_file(str(f1), vs)
    time.sleep(1)  # ensure different mtimes
    ingest_file(str(f2), vs)

    # query with path filter
    res = vs.query_advanced("alpha", k=5, where={"path": {"$contains": "a.txt"}})
    assert res["documents"][0], "should have hit for a.txt filter"

    # delete by path substring
    out = vs.delete_by_path(str(f1), exact=False)
    assert out["deleted"] >= 1

    # ensure remaining matches come from b.txt
    res2 = vs.query_advanced("alpha", k=5)
    metas = res2.get("metadatas")[0]
    assert all("b.txt" in m["path"] for m in metas)
