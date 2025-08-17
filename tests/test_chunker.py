from web.chunker import chunk_markdown

def test_chunk_basic():
    text = "# Title\n\n" + "Para. " * 200
    chunks = chunk_markdown(text, max_chars=500, overlap=50)
    assert len(chunks) >= 2
    if len(chunks) >= 2:
        assert chunks[0][-50:] in chunks[1]
