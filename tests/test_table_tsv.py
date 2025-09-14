from pathlib import Path
import importlib

try:
    tbl = importlib.import_module("local_rag_notebook.table")
except ModuleNotFoundError:
    tbl = importlib.import_module("table")


def test_tsv_auto(tmp_path: Path):
    p = tmp_path / "demo.tsv"
    p.write_text("A\tB\tC\n1\t2\t3\nx\ty\tz\n", encoding="utf-8")
    chunks = tbl.extract_csv_tsv_chunks(p, rows_per_chunk=10)
    assert len(chunks) == 1
    t = chunks[0]["text"]
    assert "| A | B | C |" in t
    assert "| 1 | 2 | 3 |" in t
    assert chunks[0]["meta"]["row_from"] == 1
    assert chunks[0]["meta"]["row_to"] == 2
