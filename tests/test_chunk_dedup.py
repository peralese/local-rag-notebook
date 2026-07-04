import pytest

from local_rag_notebook.ingest.normalize import _detok, _simple_tokenize, chunk_sections


def _section(text: str, page_no: int = 1) -> dict:
    return {
        "doc_id": "doc.pdf",
        "heading_path": "doc.pdf > Page 1",
        "page_no": page_no,
        "text": text,
        "meta": {"file_path": "doc.pdf"},
    }


def test_chunk_sections_no_duplicate_when_text_fits_one_window():
    text = "one two three four five"  # 5 tokens, well under window
    chunks = chunk_sections(_section(text), window_tokens=700, overlap_tokens=90)

    assert len(chunks) == 1
    assert chunks[0].level == "section"
    assert chunks[0].text == text


def test_chunk_sections_splits_when_text_exceeds_window():
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunk_sections(_section(text), window_tokens=8, overlap_tokens=2)

    levels = [c.level for c in chunks]
    assert levels[0] == "section"
    assert "chunk" in levels
    assert len(chunks) > 1


@pytest.mark.parametrize(
    "text",
    [
        "Task Statement 1.1: Ingest and store data.",
        "Data formats (for example, validated and non-validated formats)",
        "Experience with CI/CD pipelines and infrastructure as code (IaC)",
        "Amazon S3, Amazon EFS, and Amazon RDS are AWS services.",
    ],
)
def test_tokenize_detokenize_roundtrip_preserves_punctuation(text):
    assert _detok(_simple_tokenize(text)) == text
