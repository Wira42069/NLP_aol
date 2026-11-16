"""Dataset loading, Fuzzy Deduplication, and Robustness CV splitting (Phase 0-1).

Run once after placing the PRDECT-ID CSV in ``data/raw/``::

    python src/data_loader.py                     # auto-detects the single CSV in data/raw/
    python src/data_loader.py --csv data/raw/PRDECT-ID.csv

Outputs
-------
data/processed/dataset.csv   Fuzzy de-duplicated dataset, with a typo-injected robustness test set
data/processed/split.json    Row ids for 5-Fold CV and the holdout test set
data/processed/labels.json   Fixed label order shared by the pipeline
results/eda_summary.json     Distribution and length statistics
results/eda_*.png            Figures for the report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SEED = 42
random.seed(SEED)
# Fixed label order used by every model
EMOTIONS = ["anger", "fear", "happy", "love", "sadness"]
COLUMN_ALIASES = {
    "text": ["customer review", "customer_review", "review", "text", "ulasan"],
    "emotion": ["emotion", "emosi"],
    "sentiment": ["sentiment", "sentimen"],
}

EMOTIONS = ["anger", "fear", "happy", "love", "sadness"]
POSITIVE_EMOTIONS = {"happy", "love"}
NEGATIVE_EMOTIONS = {"anger", "fear", "sadness"}


# --------------------------------------------------------------------------- #
# Loading & Fuzzy Deduplication
# --------------------------------------------------------------------------- #
def find_raw_csv(csv_arg: str | None = None) -> Path:
    if csv_arg:
        path = Path(csv_arg)
        if not path.is_absolute():
            path = ROOT / path if (ROOT / path).exists() else Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        return path
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. Download PRDECT-ID manually and place the CSV there."
        )
    return candidates[0]


def _read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path} with utf-8 or latin-1.")


def fuzzy_deduplicate(df: pd.DataFrame, threshold: float = 0.85) -> tuple[pd.DataFrame, int]:
    """Uses MinHash LSH to drop near-duplicate reviews (e.g. seller automated responses)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    to_keep = []
    dup_count = 0
    
    for idx, text in zip(df.index, df["text"]):
        m = MinHash(num_perm=128)
        for word in str(text).lower().split():
            m.update(word.encode('utf8'))
            
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            to_keep.append(idx)
        else:
            dup_count += 1
            
    return df.loc[to_keep].reset_index(drop=True), dup_count


def load_and_clean_raw(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = _read_csv_robust(csv_path)
    
    # Resolve columns
    lookup = {str(c).strip().lower(): c for c in raw.columns}
    cols = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                cols[canon] = lookup[alias]
                break
                
    df = pd.DataFrame({"text": raw[cols["text"]], "emotion": raw[cols["emotion"]]})
    df["sentiment"] = raw[cols["sentiment"]] if "sentiment" in cols else np.nan

    stats = {"source_file": csv_path.name, "n_rows_raw": len(raw)}

    df = df.dropna(subset=["text", "emotion"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    df = df[df["text"] != ""]

    before = len(df)
    # Apply datasketch fuzzy deduplication instead of exact drops
    df, n_fuzz_dups = fuzzy_deduplicate(df)
    
    stats["n_fuzzy_duplicates_dropped"] = n_fuzz_dups
    stats["n_exact_duplicates_dropped"] = before - len(df) - n_fuzz_dups
    
    df.insert(0, "id", np.arange(len(df)))
    stats["n_rows_final"] = len(df)
    
    return df, stats


# --------------------------------------------------------------------------- #
# Robustness Splitter & Typos Injector
# --------------------------------------------------------------------------- #
def inject_indonesian_typos(text: str, noise_level: float = 0.2) -> str:
    """Simulates real-world marketplace noise by dropping vowels or repeating characters."""
    words = text.split()
    noisy_words = []
    
    for w in words:
        if random.random() < noise_level:
            if random.random() < 0.5:
                # Drop a random vowel
                vowels = [c for c in w if c in 'aiueoAIUEO']
                if vowels:
                    w = w.replace(random.choice(vowels), "", 1)
            else:
                # Duplicate a random character
                if w:
                    char = random.choice(w)
                    w = w.replace(char, char * random.choice([2, 3]), 1)
        noisy_words.append(w)
    return " ".join(noisy_words)


def make_split(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Generates a 10% test set, and 5 Stratified Folds for the remaining 90%."""
    ids = df["id"].to_numpy()
    y = df["emotion"].to_numpy()
    
    # Isolate 10% for the final Robustness Test Set
    train_val_ids, test_ids, y_train_val, _ = train_test_split(
        ids, y, test_size=0.10, stratify=y, random_state=seed
    )
    
    # 5-Fold Stratified CV on the remaining 90%
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    
    for train_idx, val_idx in skf.split(train_val_ids, y_train_val):
        folds.append({
            "train": sorted([int(train_val_ids[i]) for i in train_idx]),
            "val": sorted([int(train_val_ids[i]) for i in val_idx])
        })
        
    # We expose Fold 0 at the top level so legacy loaders (like train_baseline) 
    # can run seamlessly out of the box without needing to write a CV loop.
    return {
        "seed": seed,
        "stratified_on": "emotion",
        "n": {"train": len(folds[0]["train"]), "val": len(folds[0]["val"]), "test": len(test_ids)},
        "train": folds[0]["train"],
        "val": folds[0]["val"],
        "test": sorted(int(i) for i in test_ids),
        "cv_folds": folds
    }


def load_dataset_and_split() -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    """Shared loader used by other scripts to load data and subsets."""
    ds_path, split_path = PROC_DIR / "dataset.csv", PROC_DIR / "split.json"
    if not ds_path.exists() or not split_path.exists():
        raise FileNotFoundError("Run `python src/data_loader.py` first.")
    df = pd.read_csv(ds_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    subsets = {name: df[df["id"].isin(split[name])].reset_index(drop=True) for name in ("train", "val", "test")}
    return df, split, subsets


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, split: dict, raw_stats: dict) -> dict:
    counts = df["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
    
    return {
        "raw_file_checks": raw_stats,
        "n_reviews": len(df),
        "label_distribution_all": {
            "counts": {k: int(v) for k, v in counts.items()},
            "percent": {k: round(100 * v / max(counts.sum(), 1), 2) for k, v in counts.items()},
        },
        "length_words_mean": round(float(df["text"].str.split().str.len().mean()), 2)
    }


def save_eda_figures(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(exist_ok=True)
    counts = df["emotion"].value_counts().reindex(EMOTIONS)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.bar_label(bars)
    ax.set_title("Emotion label distribution")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eda_label_distribution.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Path to raw CSV")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite existing split")
    args = parser.parse_args(argv)

    split_path = PROC_DIR / "split.json"
    if split_path.exists() and not args.force:
        print(f"{split_path} already exists. Re-run with --force to overwrite.")
        return 1

    csv_path = find_raw_csv(args.csv)
    df, raw_stats = load_and_clean_raw(csv_path)
    split = make_split(df, args.seed)

    # Apply Typo/Noise Injection ONLY to the Test Set for Robustness Evaluation
    df["text_noisy"] = df["text"]
    test_mask = df["id"].isin(split["test"])
    df.loc[test_mask, "text_noisy"] = df.loc[test_mask, "text"].apply(
        lambda t: inject_indonesian_typos(t, noise_level=0.2)
    )
    
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(PROC_DIR / "dataset.csv", index=False)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    (PROC_DIR / "labels.json").write_text(json.dumps(EMOTIONS), encoding="utf-8")

    summary = run_eda(df, split, raw_stats)
    (RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_eda_figures(df)

    print(f"Loaded {raw_stats['n_rows_raw']} rows -> {raw_stats['n_rows_final']} after fuzzy deduplication "
          f"(Dropped {raw_stats['n_fuzzy_duplicates_dropped']} near-duplicates).")
    print(f"Robustness Test Set configured with synthetic typos. Train/Val set ready.")
    print("Wrote data/processed/{dataset.csv,split.json,labels.json}")
    return 0

if __name__ == "__main__":
    sys.exit(main())