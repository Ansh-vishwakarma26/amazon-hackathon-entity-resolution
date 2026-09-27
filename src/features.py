"""
features.py
Similarity feature computation for Source1 <-> candidate pairs.

Uses rapidfuzz when available (fast, C-accelerated -- install it on your
real machine via requirements.txt) and falls back to stdlib difflib
otherwise, so the exact same code runs in a bare-Python environment too.
Results differ slightly between the two (different algorithms under the
hood) but both are valid ratio-similarity measures -- train and predict
with the SAME one for consistency (rapidfuzz on your machine, always).
"""

import difflib
try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAVE_RAPIDFUZZ = True
except ImportError:
    _HAVE_RAPIDFUZZ = False

from normalize import name_tokens, address_tokens

FEATURE_NAMES = [
    "name_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaccard", "name_len_diff_ratio", "name_compact_containment",
    "name_tfidf_cosine",
    "addr_ratio", "addr_jaccard", "addr_numeric_jaccard",
    "addr_tfidf_cosine",
    "country_match", "country_known_both",
    "first_char_match",
]


def _ratio(a: str, b: str) -> float:
    """0-1 scale Levenshtein-based similarity ratio."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.ratio(a, b) / 100.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _token_sort_ratio(a: str, b: str) -> float:
    """Ratio after sorting each string's tokens -- neutralizes word-order
    swaps like 'Schubert Adr LLC' vs 'Schubert LLC Adr'."""
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    sa = " ".join(sorted(a.split(" ")))
    sb = " ".join(sorted(b.split(" ")))
    return _ratio(sa, sb)


def _token_set_ratio(a: str, b: str) -> float:
    """Ratio based on set operations on tokens -- robust to one string
    having extra tokens the other lacks (e.g. a trailing legal suffix)."""
    if _HAVE_RAPIDFUZZ:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0
    ta, tb = set(a.split(" ")), set(b.split(" "))
    common = ta & tb
    diff_a = ta - tb
    diff_b = tb - ta
    base = " ".join(sorted(common))
    sa = (base + " " + " ".join(sorted(diff_a))).strip()
    sb = (base + " " + " ".join(sorted(diff_b))).strip()
    return max(_ratio(base, sa), _ratio(base, sb), _ratio(sa, sb))


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _compact_containment(a: str, b: str) -> float:
    """Catches domain-style names like 'tristateguild.com' -> 'tristateguild'
    vs 'tri state guild' -> 'tristateguild' once spaces are stripped: they
    become identical/near-identical compact strings even though token
    Jaccard sees zero overlap (no shared whitespace-delimited tokens)."""
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if shorter in longer:
        return len(shorter) / len(longer)
    return _ratio(ca, cb)


def _numeric_tokens(text: str) -> frozenset:
    return frozenset(t for t in text.split(" ") if t.isdigit())


class Featurizer:
    """Holds fitted TF-IDF vectorizers (char n-gram, script-agnostic) so
    the same fitted vocabulary/IDF weights are used consistently across
    training and inference. Fit ONCE on a large text sample, then reused
    for every pair -- refitting per-pair would be both wrong (inconsistent
    scale) and far too slow at this data volume."""

    def __init__(self, name_vectorizer=None, addr_vectorizer=None):
        self.name_vectorizer = name_vectorizer
        self.addr_vectorizer = addr_vectorizer

    @staticmethod
    def fit(name_corpus: list, addr_corpus: list, max_features: int = 30000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # char_wb n-grams: works across scripts (Latin, Devanagari, Tamil...)
        # unlike word-token TF-IDF, and is naturally typo-tolerant.
        name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    max_features=max_features, lowercase=False)
        addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    max_features=max_features, lowercase=False)
        name_vec.fit([t for t in name_corpus if t])
        addr_vec.fit([t for t in addr_corpus if t])
        return Featurizer(name_vec, addr_vec)

    def _tfidf_cosine(self, vectorizer, a: str, b: str) -> float:
        if not a or not b or vectorizer is None:
            return 0.0
        va, vb = vectorizer.transform([a, b])
        num = (va.multiply(vb)).sum()
        denom = (va.multiply(va)).sum() ** 0.5 * (vb.multiply(vb)).sum() ** 0.5
        return float(num / denom) if denom > 0 else 0.0

    def compute(self, s1_name: str, s1_addr: str, s1_country: str,
                cand_name: str, cand_addr: str, cand_country: str) -> dict:
        s1_name_tok, cand_name_tok = name_tokens(s1_name), name_tokens(cand_name)
        s1_addr_tok, cand_addr_tok = address_tokens(s1_addr), address_tokens(cand_addr)

        len_a, len_b = len(s1_name), len(cand_name)
        len_diff_ratio = 1 - (abs(len_a - len_b) / max(len_a, len_b)) if max(len_a, len_b) else 1.0

        s1_country_norm = (s1_country or "").strip().lower()
        cand_country_norm = (cand_country or "").strip().lower()

        return {
            "name_ratio": _ratio(s1_name, cand_name),
            "name_token_sort_ratio": _token_sort_ratio(s1_name, cand_name),
            "name_token_set_ratio": _token_set_ratio(s1_name, cand_name),
            "name_jaccard": _jaccard(s1_name_tok, cand_name_tok),
            "name_len_diff_ratio": len_diff_ratio,
            "name_compact_containment": _compact_containment(s1_name, cand_name),
            "name_tfidf_cosine": self._tfidf_cosine(self.name_vectorizer, s1_name, cand_name),
            "addr_ratio": _ratio(s1_addr, cand_addr),
            "addr_jaccard": _jaccard(s1_addr_tok, cand_addr_tok),
            "addr_numeric_jaccard": _jaccard(_numeric_tokens(s1_addr), _numeric_tokens(cand_addr)),
            "addr_tfidf_cosine": self._tfidf_cosine(self.addr_vectorizer, s1_addr, cand_addr),
            "country_match": float(bool(s1_country_norm) and s1_country_norm == cand_country_norm),
            "country_known_both": float(bool(s1_country_norm) and bool(cand_country_norm)),
            "first_char_match": float(bool(s1_name) and bool(cand_name) and s1_name[0] == cand_name[0]),
        }

    def compute_batch(self, pairs: list) -> "list[dict]":
        """Efficient batched version for real data volumes. `pairs` is a
        list of (s1_name, s1_addr, s1_country, cand_name, cand_addr,
        cand_country) tuples. TF-IDF cosine is computed by transforming ALL
        texts in ONE vectorizer call each (not one call per pair -- that
        distinction is the difference between minutes and hours at
        hundred-thousand-plus pair volumes)."""
        import numpy as np
        n = len(pairs)
        results = [None] * n

        # cheap per-pair features (pure python; fast enough even in a loop --
        # the expensive part was always TF-IDF transform, not these)
        for i, (s1n, s1a, s1c, cn, ca, cc) in enumerate(pairs):
            s1_name_tok, cand_name_tok = name_tokens(s1n), name_tokens(cn)
            s1_addr_tok, cand_addr_tok = address_tokens(s1a), address_tokens(ca)
            len_a, len_b = len(s1n), len(cn)
            len_diff_ratio = 1 - (abs(len_a - len_b) / max(len_a, len_b)) if max(len_a, len_b) else 1.0
            s1c_n, cc_n = (s1c or "").strip().lower(), (cc or "").strip().lower()
            results[i] = {
                "name_ratio": _ratio(s1n, cn),
                "name_token_sort_ratio": _token_sort_ratio(s1n, cn),
                "name_token_set_ratio": _token_set_ratio(s1n, cn),
                "name_jaccard": _jaccard(s1_name_tok, cand_name_tok),
                "name_len_diff_ratio": len_diff_ratio,
                "name_compact_containment": _compact_containment(s1n, cn),
                "addr_ratio": _ratio(s1a, ca),
                "addr_jaccard": _jaccard(s1_addr_tok, cand_addr_tok),
                "addr_numeric_jaccard": _jaccard(_numeric_tokens(s1a), _numeric_tokens(ca)),
                "country_match": float(bool(s1c_n) and s1c_n == cc_n),
                "country_known_both": float(bool(s1c_n) and bool(cc_n)),
                "first_char_match": float(bool(s1n) and bool(cn) and s1n[0] == cn[0]),
            }

        # batched TF-IDF cosine: ONE transform call for all S1 names, ONE
        # for all candidate names, then row-aligned sparse dot products.
        s1_names = [p[0] for p in pairs]
        cand_names = [p[3] for p in pairs]
        s1_addrs = [p[1] for p in pairs]
        cand_addrs = [p[4] for p in pairs]

        name_cos = self._batch_cosine(self.name_vectorizer, s1_names, cand_names)
        addr_cos = self._batch_cosine(self.addr_vectorizer, s1_addrs, cand_addrs)
        for i in range(n):
            results[i]["name_tfidf_cosine"] = name_cos[i]
            results[i]["addr_tfidf_cosine"] = addr_cos[i]
        return results

    def _batch_cosine(self, vectorizer, texts_a: list, texts_b: list):
        import numpy as np
        if vectorizer is None:
            return [0.0] * len(texts_a)
        mat_a = vectorizer.transform(texts_a)
        mat_b = vectorizer.transform(texts_b)
        # row-aligned dot product (pair i with pair i), not full N x N matrix
        num = np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()
        norm_a = np.sqrt(np.asarray(mat_a.multiply(mat_a).sum(axis=1)).ravel())
        norm_b = np.sqrt(np.asarray(mat_b.multiply(mat_b).sum(axis=1)).ravel())
        denom = norm_a * norm_b
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(denom > 0, num / denom, 0.0)
        return cos.tolist()

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"name_vectorizer": self.name_vectorizer,
                         "addr_vectorizer": self.addr_vectorizer}, f)

    @staticmethod
    def load(path: str):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        return Featurizer(d["name_vectorizer"], d["addr_vectorizer"])
