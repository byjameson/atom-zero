"""Lightweight Turkish morphology for local retrieval.

This is intentionally not a full morphological analyzer. It handles the cases
that matter most for exact-term search: common inflectional suffixes, final
consonant mutation (``kitap`` / ``kitabı``, ``ağaç`` / ``ağacı``) and short
roots such as ``ev`` / ``evler``. It never changes indexed text; it only
normalizes terms for matching and FTS query expansion.
"""
from __future__ import annotations

import re

TURKISH_CHARS = frozenset("çğıöşüÇĞİÖŞÜ")
TURKISH_STOPWORDS = frozenset({
    "ama", "bir", "bu", "çok", "da", "daha", "de", "değil", "en", "fakat",
    "gibi", "hangi", "her", "için", "ile", "ise", "kadar", "kaç", "kim",
    "mi", "mı", "mu", "mü", "nasıl", "ne", "neden", "nedir", "nerede",
    "olan", "olarak", "önce", "şey", "şu", "sonra", "var", "ve", "yok",
})
# Detection words must not collide with common English words such as "her"
# or "var" in code snippets. Turkish-specific letters cover the rest.
TURKISH_HINTS = frozenset({
    "ama", "bir", "bu", "çok", "değil", "fakat", "gibi", "hangi", "için",
    "ile", "ise", "kadar", "kaç", "kim", "mi", "mı", "mu", "mü", "nasıl",
    "neden", "nedir", "nerede", "olan", "olarak", "önce", "şey", "şu",
    "sonra", "yok",
})
_SUFFIXES = tuple(sorted({
    "larında", "lerinde", "larından", "lerinden", "larının", "lerinin",
    "lardan", "lerden", "larda", "lerde", "ların", "lerin", "ları", "leri",
    "lar", "ler",
    "ndaki", "ndeki", "daki", "deki", "taki", "teki",
    "ndan", "nden", "dan", "den", "tan", "ten",
    "nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
    "nı", "ni", "nu", "nü", "ı", "i", "u", "ü",
    "ya", "ye", "yu", "yü", "yı", "yi",
    "yla", "yle", "la", "le",
    "lık", "lik", "luk", "lük", "lı", "li", "lu", "lü",
    "sı", "si", "su", "sü", "sız", "siz", "suz", "süz",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
    "mış", "miş", "muş", "müş", "acak", "ecek", "mak", "mek",
    "madan", "meden", "arak", "erek", "ıyor", "iyor", "uyor", "üyor",
}, key=len, reverse=True))
_FINAL_MUTATION = str.maketrans({
    "b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"})
_FINAL_ALTERNATIVES = {
    "p": ("b",), "ç": ("c",), "t": ("d",), "k": ("g", "ğ")}
# Common Turkish nouns that drop their final stem vowel before a vowel-initial
# suffix (şehir -> şehri, burun -> burnu). This is a curated list on purpose:
# applying vowel drop to every CVC word would turn kitap into "kitp".
VOWEL_DROP = {
    "şehir": "şehr", "burun": "burn", "ağız": "ağz", "oğul": "oğl",
    "isim": "ism", "resim": "resm", "gönül": "gönl", "karın": "karn",
    "boyun": "boyn", "asıl": "asl", "akıl": "akl", "fikir": "fikr",
    "ömür": "ömr", "sabır": "sabr", "nehir": "nehr", "zehir": "zehr",
    "şekil": "şekl", "kayıt": "kayd", "vakit": "vakt", "metin": "metn",
    "kavim": "kavm",
}
_DROPPED_STEMS = frozenset(VOWEL_DROP.values())
_SINGLE_VOWELS = frozenset("aeıiuü")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def turkish_lower(text: str) -> str:
    """Lowercase Turkish text without splitting the capital dotted İ."""
    return str(text or "").replace("İ", "i").replace("I", "i").lower()


def _is_suffix_tail(tail: str) -> bool:
    if not tail:
        return True
    if tail in _SINGLE_VOWELS:
        return True
    return any(tail == suffix or tail.startswith(suffix)
               or suffix.startswith(tail) for suffix in _SUFFIXES)


def looks_turkish(text: str) -> bool:
    """Return True when text has Turkish letters or common Turkish words."""
    value = str(text or "")
    if any(char in TURKISH_CHARS for char in value):
        return True
    return any(word in TURKISH_HINTS or word in VOWEL_DROP
               for word in _WORD_RE.findall(turkish_lower(value)))


def skeleton(word: str) -> str:
    """Reduce a Turkish word to a conservative matching skeleton."""
    value = turkish_lower(word).strip().replace("i̇", "i")
    if not value:
        return ""
    for full, dropped in VOWEL_DROP.items():
        if value == full:
            value = dropped
            break
        if value.startswith(dropped) and _is_suffix_tail(
                value[len(dropped):]):
            value = dropped
            break
    for _ in range(2):
        for suffix in _SUFFIXES:
            if not value.endswith(suffix):
                continue
            remaining = len(value) - len(suffix)
            minimum = 4 if len(suffix) <= 2 else 2
            if remaining < minimum:
                continue
            value = value[:-len(suffix)]
            break
        else:
            break
    if value in VOWEL_DROP:
        value = VOWEL_DROP[value]
    if value and value not in _DROPPED_STEMS:
        value = value[:-1] + value[-1].translate(_FINAL_MUTATION)
    return value


def variants(word: str) -> list[str]:
    """Return exact and prefix-friendly forms for an FTS OR query."""
    original = turkish_lower(word).strip()
    root = skeleton(original)
    stems = {original, root}
    if root:
        for alternative in _FINAL_ALTERNATIVES.get(root[-1], ()):
            stems.add(root[:-1] + alternative)
    return sorted(stem for stem in stems if stem)


def strip_query_stopwords(words) -> list[str]:
    """Drop Turkish function words from an already tokenized query."""
    return [word for word in words
            if word.lower() not in TURKISH_STOPWORDS]
