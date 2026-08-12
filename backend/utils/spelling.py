"""
Spell-check the user's query and offer a corrected version as a
*suggestion* (returned alongside the answer), not a forced rewrite -
matching the reconstructed architecture's approach.
"""
import re
from spellchecker import SpellChecker
from utils.logger import get_logger

logger = get_logger(__name__)

_spell = SpellChecker()

_WORD_RE = re.compile(r"[A-Za-z']+")


def suggest_correction(query: str) -> str | None:
    words = _WORD_RE.findall(query)
    if not words:
        return None

    misspelled = _spell.unknown([w.lower() for w in words])
    if not misspelled:
        return None

    corrected = query
    changed = False
    for word in words:
        lower = word.lower()
        if lower in misspelled:
            suggestion = _spell.correction(lower)
            if suggestion and suggestion != lower:
                corrected = re.sub(rf"\b{re.escape(word)}\b", suggestion, corrected, count=1)
                changed = True

    return corrected if changed else None
