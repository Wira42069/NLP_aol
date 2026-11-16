"""Toggleable text preprocessing pipeline (Phase 1).

Steps (each can be switched on/off through :class:`PreprocessConfig`):
    1. lowercase
    2. clean            - remove URLs/HTML entities/mentions, collapse repeated characters and
                          punctuation, drop special characters (emoji optionally kept)
    3. slang            - word-level normalisation with ``new_kamusalay.csv``
    4. stopwords        - Indonesian stopword removal that PRESERVES negations and intensifiers

Stemming (Sastrawi) is intentionally skipped (core rule 4).

Presets used for the ablation study:
    raw                      nothing applied
    cleaned                  lowercase + clean
    cleaned_slang            lowercase + clean + slang        <- default for IndoBERT
    cleaned_slang_stopwords  ... + stopword removal           <- baseline-only extra condition

Stopword removal is only ever meant for the TF-IDF baseline. Do NOT feed stopword-stripped text
to IndoBERT: a transformer needs the full sentence structure.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_SLANG_CSV = ROOT / "data" / "external" / "new_kamusalay.csv"
SAVED_SLANG_JSON = ROOT / "models" / "slang_dict.json"

# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
# Words that carry polarity / emotion intensity and must survive stopword removal.
NEGATIONS = {"tidak", "nggak", "ngga", "gak", "ga", "tak", "kurang", "jangan", "bukan", "belum", "tdk", "enggak"}
INTENSIFIERS = {"sangat", "sekali", "banget", "terlalu", "paling", "lebih", "cukup", "agak", "hanya", "cuma", "tapi", "namun", "sayang"}

_BASE_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada", "adalah", "ada", "atau",
    "juga", "karena", "oleh", "dalam", "akan", "saya", "aku", "kami", "kita", "kamu", "anda", "dia", "mereka",
    "nya", "sudah", "telah", "sedang", "masih", "bisa", "dapat", "harus", "para", "sebuah", "suatu", "saat",
    "ketika", "jika", "kalau", "bila", "sebagai", "agar", "supaya", "sehingga", "lalu", "kemudian", "setelah",
    "sebelum", "hingga", "sampai", "antara", "tentang", "terhadap", "dalam", "atas", "bawah", "se", "pun",
    "lah", "kah", "dong", "sih", "deh", "nih", "tuh", "ya", "yah", "oh", "eh", "kok", "lho", "loh",
    "mau", "ingin", "sama", "semua", "setiap", "setiap", "banyak", "beberapa", "lain", "lainnya", "hal",
    "yaitu", "yakni", "ialah", "serta", "maupun", "bahwa", "tersebut", "begitu", "sebagai", "dll", "dsb",
    "dgn", "yg", "dr", "utk", "krn", "jd", "jadi", "udah", "udh", "sdh", "lg", "lagi", "aja", "saja", "sih",
}
STOPWORDS = frozenset(_BASE_STOPWORDS - NEGATIONS - INTENSIFIERS)

# Tiny built-in fallback used only when new_kamusalay.csv is not available.
FALLBACK_SLANG = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak", "enggak": "tidak", "tdk": "tidak",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "bgt": "banget", "bngt": "banget",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "jd": "jadi", "tp": "tapi",
    "bgs": "bagus", "mantul": "mantap betul", "mantab": "mantap", "recomended": "rekomendasi",
    "recommended": "rekomendasi", "rekomen": "rekomendasi", "brg": "barang", "barangnya": "barang nya",
    "pengirimannya": "pengiriman nya", "thx": "terima kasih", "thanks": "terima kasih", "makasih": "terima kasih",
    "trims": "terima kasih", "kcw": "kecewa", "kecewaa": "kecewa", "lmyn": "lumayan", "sy": "saya",
    "dr": "dari", "sm": "sama", "jgn": "jangan", "klo": "kalau", "kalo": "kalau", "emg": "memang",
    "bkn": "bukan", "lbh": "lebih", "trs": "terus", "skrg": "sekarang", "hrg": "harga", "hrga": "harga",
    "cpt": "cepat", "lama2": "lama", "sampe": "sampai", "sampai2": "sampai",
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_ENTITY_RE = re.compile(r"&[a-z]+;|&#\d+;", re.IGNORECASE)
_MENTION_RE = re.compile(r"[@#]\w+")
_REPEAT_CHAR_RE = re.compile(r"([a-z])\1{2,}")  # "bagusssss" -> "bagus"
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐❤]")
_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Slang dictionary
# --------------------------------------------------------------------------- #
def load_kamusalay(path: Path) -> dict[str, str]:
    """Read new_kamusalay.csv (two columns, no header, latin-1) into a dict."""
    import pandas as pd

    df = pd.read_csv(path, header=None, names=["slang", "formal"], encoding="latin-1")
    df = df.dropna()
    return {str(s).strip().lower(): str(f).strip().lower() for s, f in zip(df["slang"], df["formal"])}


def resolve_slang_dict(path: str | Path | None = None) -> tuple[dict[str, str], str]:
    """Return (dictionary, source). Order: explicit path > models/slang_dict.json > data/external/ > fallback."""
    if path:
        p = Path(path)
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8")), f"json:{p.name}"
        return load_kamusalay(p), f"kamusalay:{p.name}"
    if SAVED_SLANG_JSON.exists():
        return json.loads(SAVED_SLANG_JSON.read_text(encoding="utf-8")), "json:models/slang_dict.json"
    if EXTERNAL_SLANG_CSV.exists():
        return load_kamusalay(EXTERNAL_SLANG_CSV), "kamusalay:data/external/new_kamusalay.csv"
    warnings.warn(
        "new_kamusalay.csv not found in data/external/ - using the tiny built-in fallback slang list. "
        "Download the real file for proper results (see README step 2).",
        stacklevel=2,
    )
    return dict(FALLBACK_SLANG), "builtin-fallback"


# --------------------------------------------------------------------------- #
# Config + preprocessor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    clean: bool = True
    slang: bool = True
    stopwords: bool = False
    keep_emoji: bool = False  # brief says drop special chars; flip to True to keep emoji as tokens

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


PRESETS: dict[str, PreprocessConfig] = {
    "raw": PreprocessConfig(lowercase=False, clean=False, slang=False, stopwords=False),
    "cleaned": PreprocessConfig(lowercase=True, clean=True, slang=False, stopwords=False),
    "cleaned_slang": PreprocessConfig(lowercase=True, clean=True, slang=True, stopwords=False),
    "cleaned_slang_stopwords": PreprocessConfig(lowercase=True, clean=True, slang=True, stopwords=True),
}
# The three conditions required by the brief's ablation (the 4th is a baseline-only extra).
BRIEF_ABLATION_PRESETS = ["raw", "cleaned", "cleaned_slang"]


class Preprocessor:
    """Stateless text normaliser; safe to apply independently to train/val/test."""

    def __init__(self, config: PreprocessConfig | str = "cleaned_slang", slang_dict: dict[str, str] | None = None):
        self.config = PRESETS[config] if isinstance(config, str) else config
        self.slang_dict = slang_dict or {}
        if self.config.slang and not self.slang_dict:
            raise ValueError("Slang normalisation is enabled but no slang dictionary was supplied.")

    # -- individual steps ------------------------------------------------- #
    def _clean(self, text: str) -> str:
        text = _URL_RE.sub(" ", text)
        text = _ENTITY_RE.sub(" ", text)
        text = _MENTION_RE.sub(" ", text)
        text = _REPEAT_CHAR_RE.sub(r"\1", text)
        if self.config.keep_emoji:
            text = _EMOJI_RE.sub(lambda m: f" {m.group(0)} ", text)
            text = re.sub(r"[^\w\s\U0001F300-\U0001FAFF☀-➿⭐❤]", " ", text)
        else:
            text = re.sub(r"[^\w\s]", " ", text)  # punctuation, symbols, emoji -> space
        text = text.replace("_", " ")
        return text

    def _normalise_slang(self, text: str) -> str:
        d = self.slang_dict
        return " ".join(d.get(tok, tok) for tok in text.split())

    @staticmethod
    def _remove_stopwords(text: str) -> str:
        return " ".join(tok for tok in text.split() if tok not in STOPWORDS)

    # -- public API -------------------------------------------------------- #
    def __call__(self, text: object) -> str:
        if text is None or (isinstance(text, float) and text != text):
            return ""
        text = str(text)
        cfg = self.config
        if cfg.lowercase:
            text = text.lower()
        if cfg.clean:
            text = self._clean(text)
        if cfg.slang:
            text = self._normalise_slang(text)
        if cfg.stopwords:
            text = self._remove_stopwords(text)
        return _WS_RE.sub(" ", text).strip()

    def transform(self, texts) -> list[str]:
        return [self(t) for t in texts]

    def count_slang_hits(self, text: str) -> int:
        """Number of tokens the slang dictionary would rewrite (used for error-analysis hints)."""
        base = text.lower()
        base = self._clean(base) if self.config.clean else base
        return sum(1 for tok in base.split() if tok in self.slang_dict and self.slang_dict[tok] != tok)


def save_preprocess_meta(path: Path, preset_name: str, config: PreprocessConfig, slang_source: str) -> None:
    path.write_text(
        json.dumps({"preset": preset_name, "config": config.to_dict(), "slang_source": slang_source}, indent=2),
        encoding="utf-8",
    )


def load_preprocessor(meta_path: Path, slang_path: Path | None = None) -> Preprocessor:
    """Rebuild the exact preprocessor a model was trained with."""
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    cfg = PreprocessConfig.from_dict(meta["config"])
    slang = None
    if cfg.slang:
        slang, _ = resolve_slang_dict(slang_path)
    return Preprocessor(cfg, slang)


if __name__ == "__main__":  # tiny demo
    sd, src = resolve_slang_dict()
    demo = "Barangnya BAGUSSSS bgt!!! gak nyesel, pengiriman cepet https://tokopedia.com/xyz 😍"
    for name, cfg in PRESETS.items():
        print(f"{name:26s}->", Preprocessor(cfg, sd)(demo))
    print("slang source:", src)
