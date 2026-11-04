"""
Persian text normalization for Digikala reviews.

Design rules (justified in the Persian report):
  1. Character unification (ي→ی, ك→ک, ة→ه, ۀ→ه).
  2. Arabic-Indic / Persian-Indic digits → ASCII digits
     (numbers carry real meaning: "۴ ستاره").
  3. ZWNJ (نیم‌فاصله) is KEPT — it is a semantic character in Persian.
  4. Kashida (ـ) removed — purely decorative.
  5. URL / HTML stripped — no sentiment signal, leaks source.
  6. Emojis KEPT — they carry strong sentiment ("👎").
  7. Diacritics (اعراب) removed — noisy in user-generated text.
  8. Whitespace collapsed.
  9. NEGATION WORDS ARE NEVER REMOVED ("نه", "نمی", "بی", "بدون").
"""
import re
import unicodedata

# Arabic-Indic and Persian-Indic digit ranges
_AR_INDIC = "٠١٢٣٤٥٦٧٨٩"
_FA_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_D  = "0123456789"
_DIGIT_MAP = str.maketrans(
    _AR_INDIC + _FA_INDIC,
    _ASCII_D * 2,
)

# Character unification
_CHAR_MAP = str.maketrans({
    "ي": "ی",   # Arabic yeh → Persian yeh
    "ك": "ک",   # Arabic kaf → Persian kaf
    "ة": "ه",   # Arabic teh marbuta → heh
    "ۀ": "ه",   # heh with yeh above → heh
    "أ": "ا",
    "إ": "ا",
    "ؤ": "و",
    "ئ": "ی",
    "ٱ": "ا",
})

# Diacritics (harakat) — zero-width combining marks
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# HTML tags
_HTML_RE = re.compile(r"<[^>]+>")

# URLs (very permissive)
_URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|t\.me/\S+|instagram\.com/\S+)",
    flags=re.IGNORECASE,
)

# Collapse whitespace (including weird unicode spaces)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)

# Zero-width non-joiner (KEPT) vs other zero-width chars (removed)
_ZWNJ = "\u200c"
_ZWJ  = "\u200d"
_BOM  = "\ufeff"
_LRM  = "\u200e"
_RLM  = "\u200f"


def normalize_fa(text: str) -> str:
    """Full normalization pipeline. Must be IDENTICAL at train and serve time."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Unicode NFC first (composes characters consistently)
    text = unicodedata.normalize("NFC", text)

    # 2. Strip zero-width noise but keep ZWNJ
    text = text.replace(_ZWJ, "").replace(_BOM, "").replace(_LRM, "").replace(_RLM, "")

    # 3. Digit unification
    text = text.translate(_DIGIT_MAP)

    # 4. Character unification
    text = text.translate(_CHAR_MAP)

    # 5. Remove diacritics + kashida
    text = _DIACRITICS_RE.sub("", text)

    # 6. HTML & URLs
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)

    # 7. Collapse whitespace (but DO NOT touch ZWNJ)
    text = _WS_RE.sub(" ", text).strip()

    return text


def is_effectively_empty(text: str, min_chars: int = 2) -> bool:
    """Drop reviews that normalize to nothing meaningful (e.g. '.....', 'ok')."""
    return len(text.replace(" ", "")) < min_chars