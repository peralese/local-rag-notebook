from __future__ import annotations


def test_import_package() -> None:
    """Basic import smoke test so CI can verify package layout."""
    import local_rag_notebook  # noqa: F401
