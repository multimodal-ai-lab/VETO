import re


def parse_yes_no(raw: str) -> bool:
    s = (raw or "").strip().lower()
    if not s:
        return False
    return re.search(r"\byes\b", s) is not None
