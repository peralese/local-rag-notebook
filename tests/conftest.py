import sys
from pathlib import Path

# Repo root = one level above tests/
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repo root is on sys.path so `import synthesizer` works (root file).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also ensure `local_rag_notebook` package (if present) is importable.
PKG_DIR = REPO_ROOT / "local_rag_notebook"
if PKG_DIR.exists() and str(REPO_ROOT) not in sys.path:
    # (Already added REPO_ROOT above; leaving here for clarity.)
    pass
