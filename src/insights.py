"""Rule-based seller-insights module (Phase 5).

Maps each review to e-commerce aspects with a keyword lexicon and combines that with the predicted
emotion to show sellers WHAT customers talk about and HOW they feel about it.

``CORE_KEYWORDS`` are exactly the keywords from the project brief; ``EXTRA_KEYWORDS`` are a few
obvious additions that raise coverage (set ``use_extra=False`` to use the brief's lexicon only).

Limitations (mention them in the report): keyword matching is context-blind ("cepat" can describe
shipping or something else), and generic praise words such as "bagus" are mapped to Product Quality.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pandas as pd

ASPECTS = ["Shipping/Delivery", "Product Quality", "Packaging", "Pricing", "Seller Service"]
NO_ASPECT = "No aspect detected"
EMOTION_ORDER = ["anger", "fear", "happy", "love", "sadness"]
NEGATIVE_EMOTIONS = ("anger", "fear", "sadness")

CORE_KEYWORDS: dict[str, list[str]] = {
    "Shipping/Delivery": ["pengiriman", "kurir", "ekspedisi", "lambat", "cepat"],
    "Product Quality": ["kualitas", "bahan", "rusak", "bagus", "awet", "cacat"],
    "Packaging": ["packing", "bungkus", "bubble wrap", "rapi", "bocor"],
    "Pricing": ["harga", "mahal", "murah", "terjangkau", "promo"],
    "Seller Service": ["respon", "seller", "admin", "ramah", "balasan"],
}
EXTRA_KEYWORDS: dict[str, list[str]] = {
    "Shipping/Delivery": ["dikirim", "kirim", "ongkir", "sampai", "telat", "paket", "jne", "jnt", "sicepat", "gosend", "resi"],
    "Product Quality": ["barang", "produk", "kualitasnya", "tipis", "kasar", "mulus", "original", "ori", "palsu", "jelek", "sesuai"],
    "Packaging": ["kemasan", "dus", "kardus", "lakban", "penyok", "pecah", "kemas", "dikemas"],
    "Pricing": ["diskon", "worth", "sepadan", "ongkos", "biaya"],
    "Seller Service": ["penjual", "toko", "pelayanan", "cs", "chat", "dibalas", "fast respon", "slow respon", "retur", "refund"],
}


@lru_cache(maxsize=4)
def _compiled(use_extra: bool) -> dict[str, re.Pattern]:
    patterns = {}
    for aspect in ASPECTS:
        words = list(CORE_KEYWORDS[aspect]) + (list(EXTRA_KEYWORDS[aspect]) if use_extra else [])
        # whole-word / whole-phrase match, longest first so "bubble wrap" wins over "bubble"
        alt = "|".join(re.escape(w) for w in sorted(set(words), key=len, reverse=True))
        patterns[aspect] = re.compile(rf"(?<!\w)(?:{alt})(?!\w)")
    return patterns


def tag_aspects(text: str, use_extra: bool = True) -> dict:
    """Return {'aspects': {aspect: [matched keywords]}, 'primary': aspect or NO_ASPECT}.

    Primary aspect = the aspect with the most keyword hits; ties go to the aspect that is
    mentioned first in the text.
    """
    text = str(text).lower()
    hits: dict[str, list[str]] = {}
    first_pos: dict[str, int] = {}
    for aspect, pat in _compiled(use_extra).items():
        matches = list(pat.finditer(text))
        if matches:
            hits[aspect] = [m.group(0) for m in matches]
            first_pos[aspect] = matches[0].start()
    if not hits:
        return {"aspects": {}, "primary": NO_ASPECT}
    primary = sorted(hits, key=lambda a: (-len(hits[a]), first_pos[a]))[0]
    return {"aspects": hits, "primary": primary}


def analyze_reviews(reviews: list[str], emotions: list[str] | None = None, use_extra: bool = True) -> pd.DataFrame:
    """Review list -> DataFrame with primary aspect, all matched aspects and (optionally) the emotion."""
    rows = []
    for i, text in enumerate(reviews):
        tagged = tag_aspects(text, use_extra)
        rows.append(
            {
                "review": text,
                "primary_aspect": tagged["primary"],
                "all_aspects": ", ".join(tagged["aspects"]) if tagged["aspects"] else "",
                "matched_keywords": ", ".join(sorted({k for v in tagged["aspects"].values() for k in v})),
                **({"emotion": emotions[i]} if emotions is not None else {}),
            }
        )
    return pd.DataFrame(rows)


def aspect_emotion_matrix(
    df: pd.DataFrame,
    use_all_aspects: bool = False,
    include_no_aspect: bool = True,
    emotion_col: str = "emotion",
) -> pd.DataFrame:
    """Counts of reviews per aspect (rows) x emotion (columns).

    ``use_all_aspects=False`` counts each review once under its primary aspect (rows sum to the
    number of reviews). ``True`` counts a review under every aspect it mentions.
    """
    if use_all_aspects:
        exploded = df.assign(aspect=df["all_aspects"].fillna("").str.split(", ")).explode("aspect")
        exploded = exploded.reset_index(drop=True)  # explode() repeats index labels, which breaks crosstab
        exploded["aspect"] = exploded["aspect"].replace("", NO_ASPECT).fillna(NO_ASPECT)
    else:
        exploded = df.assign(aspect=df["primary_aspect"])
    rows = ASPECTS + ([NO_ASPECT] if include_no_aspect else [])
    mat = pd.crosstab(exploded["aspect"], exploded[emotion_col]).reindex(index=rows, columns=EMOTION_ORDER, fill_value=0)
    return mat.fillna(0).astype(int)


def summarize_for_sellers(df: pd.DataFrame, min_reviews: int = 3, top_n: int = 3, emotion_col: str = "emotion") -> list[str]:
    """Plain-language action points computed from the data (no numbers are invented)."""
    mat = aspect_emotion_matrix(df, include_no_aspect=False, emotion_col=emotion_col)
    totals = mat.sum(axis=1)
    lines = []
    scored = []
    for aspect in mat.index:
        if totals[aspect] < min_reviews:
            continue
        neg = int(mat.loc[aspect, list(NEGATIVE_EMOTIONS)].sum())
        scored.append((neg / totals[aspect], neg, int(totals[aspect]), aspect))
    for share, neg, total, aspect in sorted(scored, reverse=True)[:top_n]:
        if neg == 0:
            continue
        worst = max(NEGATIVE_EMOTIONS, key=lambda e: mat.loc[aspect, e])
        lines.append(
            f"**{aspect}**: {share:.0%} of the {total} reviews mentioning it express a negative emotion "
            f"(most common: {worst}, {int(mat.loc[aspect, worst])} reviews). Worth checking first."
        )
    best = sorted(((int(mat.loc[a, ["happy", "love"]].sum()) / totals[a], int(totals[a]), a)
                   for a in mat.index if totals[a] >= min_reviews), reverse=True)
    if best and best[0][0] > 0:
        share, total, aspect = best[0]
        lines.append(f"**{aspect}** is your strongest point: {share:.0%} of the {total} reviews mentioning it are happy or love.")
    if not lines:
        lines.append("Not enough aspect mentions yet to highlight priorities (need at least a few reviews per aspect).")
    return lines


def keyword_counts(reviews: list[str], use_extra: bool = True, top_k: int = 10) -> pd.DataFrame:
    """Most frequent matched keywords overall (handy for the report's lexicon-coverage table)."""
    c: Counter = Counter()
    for t in reviews:
        for kws in tag_aspects(t, use_extra)["aspects"].values():
            c.update(kws)
    return pd.DataFrame(c.most_common(top_k), columns=["keyword", "count"])


if __name__ == "__main__":
    demo = [
        "Pengiriman cepat tapi packing kurang rapi",
        "Harga murah, kualitas bagus, seller ramah",
        "Barang rusak dan admin tidak ada respon",
        "Mantap",
    ]
    emo = ["happy", "sadness", "anger", "love"]
    out = analyze_reviews(demo, emo)
    print(out[["review", "primary_aspect", "all_aspects", "emotion"]].to_string(index=False))
    print(aspect_emotion_matrix(out))
