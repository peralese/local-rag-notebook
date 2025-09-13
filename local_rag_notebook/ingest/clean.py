import re

BULLETS = ["•", "◦", "‣", "▪", "▸", "►", "●", "○", "■", "□", "–", "—", "·", ""]


def normalize_text(s: str) -> str:
    if not s:
        return s
    # Normalize Windows line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Replace bullet-like characters with "- "
    for b in BULLETS:
        s = s.replace(b, "- ")
    # Fix common ligatures / oddities (optional minimal set)
    s = s.replace("\u00a0", " ")  # nbsp -> space
    # De-hyphenate line breaks like "configu-\nration" -> "configuration"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # Collapse multiple spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    # Trim excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
