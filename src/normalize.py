"""
normalize.py
Text normalization for business names and addresses.

Design notes:
- We normalize once per record and cache the result (never re-normalize the
  same string twice in a run) because normalization is called millions of
  times across blocking + feature extraction.
- We keep normalization deterministic and reversible-ish (we don't destroy
  information we might need later, e.g. we don't drop numbers).
"""

import re
import unicodedata

# --- Legal-suffix / common abbreviation expansion -------------------------
# Map noisy variants -> a single canonical token. Applied on whitespace-split
# tokens so word-order doesn't matter and "Corp" / "Corporation" collapse to
# the same thing for similarity purposes.
NAME_TOKEN_MAP = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc",
    "sarl": "sarl",
    "sas": "sas",
    "sa": "sa",
    "gmbh": "gmbh",
    "and": "and", "&": "and",
    "the": "",  # drop leading articles, they carry no signal
}

ADDRESS_TOKEN_MAP = {
    "rd": "rd", "road": "rd",
    "st": "st", "street": "st",
    "ave": "ave", "avenue": "ave",
    "blvd": "blvd", "boulevard": "blvd",
    "dr": "dr", "drive": "dr",
    "ln": "ln", "lane": "ln",
    "ct": "ct", "court": "ct",
    "pl": "pl", "place": "pl",
    "hwy": "hwy", "highway": "hwy",
    "apt": "apt", "apartment": "apt", "unit": "apt",
    "no": "no", "number": "no",
    "po": "po",
}

def _fold_latin_accents(text: str) -> str:
    """Strip accents (á->a, é->e) but ONLY for Latin-script text. Blanket
    NFKD + combining-mark removal would corrupt Devanagari/other scripts,
    where combining marks are vowel signs, not decorative accents."""
    out_chars = []
    for ch in text:
        if ch.isascii():
            out_chars.append(ch)
            continue
        # Decompose this one character (e.g. 'á' -> 'a' + combining acute)
        decomp = unicodedata.normalize("NFD", ch)
        base_char = decomp[0] if decomp else ch
        if base_char.isascii() and base_char.isalpha() and len(decomp) > 1:
            out_chars.append(base_char)  # was a Latin letter + accent mark
        else:
            out_chars.append(ch)  # non-Latin script (e.g. Devanagari): keep as-is
    return "".join(out_chars)


_WS_RE = re.compile(r"\s+")
_MULTI_PLUS_RE = re.compile(r"\++")

# Categories to KEEP as-is: L* (letters, any script), N* (digits),
# M* (combining marks -- critical for Devanagari and other scripts where
# vowel signs are combining characters, not decoration). Everything else
# (brackets, #, -, ., etc) becomes a space. Python's regex \w does NOT
# include category M, which is why we do this per-character instead of a
# blanket [^\w\s] regex -- that regex silently corrupts Devanagari text.
_KEEP_CATEGORIES = ("L", "N", "M")


def _strip_punct_unicode_safe(text: str) -> str:
    out = []
    for ch in text:
        if ch == "+" or ch.isspace():
            out.append(ch)
        elif unicodedata.category(ch)[0] in _KEEP_CATEGORIES:
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def strip_domain_suffix(name: str) -> str:
    """'tristateguild.com' -> 'tristateguild' (still won't token-match
    'Tri-State Guild' well, but downstream char n-gram / fuzzy features
    pick up the similarity; this at least removes the dead '.com' token)."""
    return re.sub(r"\.(com|net|org|co|in|io|biz)$", "", name.strip().lower())


def _base_clean(text: str) -> str:
    if text is None or (isinstance(text, float)):  # NaN
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _fold_latin_accents(text)
    text = strip_domain_suffix(text) if "." in text and " " not in text.strip() else text
    text = _strip_punct_unicode_safe(text)
    text = _MULTI_PLUS_RE.sub("+", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Normalize a business name: clean punctuation, expand/canonicalize
    legal-suffix tokens, drop empty tokens. Word order is NOT enforced here
    (that's what token-set / token-sort similarity features are for)."""
    text = _base_clean(name)
    if not text:
        return ""
    tokens = text.split(" ")
    out = []
    for tok in tokens:
        mapped = NAME_TOKEN_MAP.get(tok, tok)
        if mapped:
            out.append(mapped)
    return " ".join(out)


def normalize_address(address: str) -> str:
    """Normalize a business address: clean punctuation, canonicalize street
    abbreviations. Keeps digits (house/unit numbers matter for matching)."""
    text = _base_clean(address)
    if not text:
        return ""
    tokens = text.split(" ")
    out = [ADDRESS_TOKEN_MAP.get(tok, tok) for tok in tokens]
    return " ".join(out)


def name_tokens(normalized_name: str) -> frozenset:
    """Token set for Jaccard similarity, with near-universal legal-suffix
    tokens excluded so two different companies that are both 'X LLC' don't
    look similar just because of 'llc'."""
    STOP = {"inc", "corp", "co", "ltd", "llc", "llp", "pvt", "plc", "and", ""}
    return frozenset(t for t in normalized_name.split(" ") if t and t not in STOP)


def address_tokens(normalized_address: str) -> frozenset:
    return frozenset(t for t in normalized_address.split(" ") if t)


def strip_entity_prefix(entity_id: str) -> int:
    """'S1-965667' -> 965667. Saves substantial memory vs keeping the
    string form for millions of IDs; we re-attach the prefix on output."""
    return int(entity_id.split("-", 1)[1])
