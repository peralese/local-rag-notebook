import csv
from pathlib import Path
import importlib

try:
    tbl = importlib.import_module("local_rag_notebook.table")
except ModuleNotFoundError:
    tbl = importlib.import_module("table")  # fallback if you keep it at repo root


def _write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_read_and_chunk_csv_basic(tmp_path: Path):
    p = tmp_path / "demo.csv"
    header = ["Name", "Score", "Note"]
    rows = [[f"row{i}", str(i), f"note {i}"] for i in range(1, 51)]
    _write_csv(p, header, rows)

    chunks = tbl.extract_csv_tsv_chunks(p, rows_per_chunk=15)
    assert len(chunks) == 4
    first = chunks[0]
    assert first["meta"]["kind"] == "table"
    assert first["meta"]["row_from"] == 1
    assert first["meta"]["row_to"] == 15
    assert first["meta"]["total_rows"] == 50
    assert "| Name | Score | Note |" in first["text"]
    assert "| row1 | 1 | note 1 |" in first["text"]


def test_chunk_overlap_and_bounds(tmp_path: Path):
    p = tmp_path / "small.csv"
    header = ["A", "B"]
    rows = [[str(i), str(i * 2)] for i in range(1, 11)]
    _write_csv(p, header, rows)

    chunks = tbl.extract_csv_tsv_chunks(p, rows_per_chunk=6, overlap=2)
    assert len(chunks) == 2
    assert chunks[0]["meta"]["row_from"] == 1 and chunks[0]["meta"]["row_to"] == 6
    assert chunks[1]["meta"]["row_from"] == 5 and chunks[1]["meta"]["row_to"] == 10
    assert "| A | B |" in chunks[1]["text"]
    assert "| 10 | 20 |" in chunks[1]["text"]


def test_direct_table_helpers(tmp_path: Path):
    p = tmp_path / "raw.csv"
    header = ["Col1", "Col2", "Col3"]
    rows = [["x", "y", "z"], ["longtext"*20, "b", "c"]]
    _write_csv(p, header, rows)

    tblobj = tbl.read_csv(p)
    assert tblobj.columns == header
    ch = tbl.chunk_table(tblobj, rows_per_chunk=2)[0]
    assert ch["text"].splitlines()[0].startswith("| Col1 | Col2 | Col3 |")
    assert ch["meta"]["source_path"].endswith("raw.csv")
    assert ch["meta"]["title"] == "raw.csv"

