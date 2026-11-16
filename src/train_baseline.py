"""CPU-optimized training pipeline using Subword Feature Union, Calibrated LightGBM, and FastText Aspect Embeddings.

    python src/train_baseline.py

Outputs
-------
models/baseline.joblib                 Fitted FeatureUnion + Calibrated LightGBM Pipeline
models/mlb.joblib                      Fitted MultiLabelBinarizer
models/fasttext_aspects.model          Trained Gensim FastText model for aspect similarity
models/baseline_meta.json              Pipeline metadata, classes, aspect seeds, and metrics
results/predictions_baseline_test.csv  Calibrated probabilities across clean and noisy test sets
results/model_comparison.csv           Summary metrics table
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import EMOTIONS, PROC_DIR, RESULTS_DIR, SEED, load_dataset_and_split  # noqa: E402
from evaluate import update_model_comparison  # noqa: E402

MODELS_DIR = ROOT / "models"

# Aspect seeds used by the FastText vector engine
ASPECT_SEEDS = {
    "shipping": ["kurir", "pengiriman", "kirim", "lama", "cepat", "estimasi", "ekspedisi"],
    "quality": ["kualitas", "bagus", "rusak", "cacat", "original", "ori", "fungsi", "awet"],
    "packaging": ["buble", "bubble", "packing", "kemasan", "rapi", "dus", "box", "aman"],
    "pricing": ["harga", "murah", "mahal", "terjangkau", "diskon", "promo", "worth"],
    "service": ["penjual", "seller", "admin", "respon", "ramah", "pelayanan", "balas"]
}


def build_feature_union() -> FeatureUnion:
    """Combines word-level TF-IDF and character-level subword TF-IDF.
    
    Character n-grams (3-5) handle Indonesian slang, typos, and repeated letters 
    without needing an external dictionary.
    """
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    lowercase=True
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    lowercase=True
                )
            )
        ]
    )


def build_calibrated_pipeline() -> Pipeline:
    """Builds a LightGBM classifier wrapped in Platt Scaling calibration."""
    base_lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1
    )
    # CalibratedClassifierCV ensures raw probabilities match empirical confidence
    calibrated_clf = CalibratedClassifierCV(estimator=base_lgbm, method="sigmoid", cv=3)
    ovr_clf = OneVsRestClassifier(calibrated_clf)

    return Pipeline([
        ("features", build_feature_union()),
        ("clf", ovr_clf)
    ])


def train_fasttext_aspects(sentences: list[list[str]]) -> FastText:
    """Trains a subword-aware FastText embedding model for semantic aspect matching."""
    return FastText(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10,
        seed=SEED
    )


def main() -> int:
    t0 = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load complete dataset, splits, and predefined subsets
    df, split, sub = load_dataset_and_split()
    train, val, test = sub["train"], sub["val"], sub["test"]
    print(f"Loaded dataset ({len(df)} rows) -> Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # 1. Initialize and FIT MultiLabelBinarizer explicitly before transform calls
    mlb = MultiLabelBinarizer(classes=EMOTIONS)
    mlb.fit([EMOTIONS])

    # 2. 5-Fold Cross-Validation Loop across train+val folds
    print("\nRunning 5-Fold Cross-Validation...")
    cv_f1s, cv_accs = [], []

    for fold_idx, fold in enumerate(split["cv_folds"]):
        train_fold = df[df["id"].isin(fold["train"])]
        val_fold = df[df["id"].isin(fold["val"])]

        y_tr = mlb.transform(train_fold["emotion"].apply(lambda x: [x] if isinstance(x, str) else x))
        y_va = mlb.transform(val_fold["emotion"].apply(lambda x: [x] if isinstance(x, str) else x))

        fold_pipe = build_calibrated_pipeline()
        fold_pipe.fit(train_fold["text"], y_tr)

        val_probs = fold_pipe.predict_proba(val_fold["text"])
        val_preds = (val_probs >= 0.5).astype(int)

        f1 = f1_score(y_va, val_preds, average="macro", zero_division=0)
        acc = accuracy_score(y_va, val_preds)
        cv_f1s.append(f1)
        cv_accs.append(acc)

    val_f1 = float(np.mean(cv_f1s))
    val_acc = float(np.mean(cv_accs))
    print(f" 5-Fold CV Macro-F1: {val_f1:.4f} (+/- {np.std(cv_f1s):.4f}) | Accuracy: {val_acc:.4f}\n")

    # 3. Fit final pipeline on combined Train + Validation set (90% of data)
    print("Training final pipeline on full Train + Validation set...")
    train_val = pd.concat([train, val]).reset_index(drop=True)
    train_val_labels = mlb.transform(train_val["emotion"].apply(lambda x: [x] if isinstance(x, str) else x))

    pipeline = build_calibrated_pipeline()
    pipeline.fit(train_val["text"], train_val_labels)

    # 4. Train FastText Aspect Embeddings on full Train + Val corpus
    print("Training FastText Aspect Embeddings...")
    tokenized_corpus = [text.lower().split() for text in train_val["text"]]
    ft_model = train_fasttext_aspects(tokenized_corpus)
    ft_model.save(str(MODELS_DIR / "fasttext_aspects.model"))

    # 5. Evaluate on Clean vs. Noisy (Synthetic Typos) Test Sets
    print("Evaluating on Clean vs. Noisy Test Sets...")
    test_clean_probs = pipeline.predict_proba(test["text"])
    test_noisy_probs = pipeline.predict_proba(test["text_noisy"] if "text_noisy" in test.columns else test["text"])

    clean_preds = [EMOTIONS[i] for i in test_clean_probs.argmax(axis=1)]
    noisy_preds = [EMOTIONS[i] for i in test_noisy_probs.argmax(axis=1)]

    clean_f1 = f1_score(test["emotion"], clean_preds, average="macro")
    noisy_f1 = f1_score(test["emotion"], noisy_preds, average="macro")

    print(f" Test Set Macro-F1 (Clean): {clean_f1:.4f}")
    print(f" Test Set Macro-F1 (Noisy): {noisy_f1:.4f} (Robustness Drop: {clean_f1 - noisy_f1:.4f})")

    # Export prediction outputs
    out = pd.DataFrame({
        "id": test["id"],
        "text": test["text"],
        "text_noisy": test["text_noisy"] if "text_noisy" in test.columns else test["text"],
        "true": test["emotion"],
        "pred": clean_preds,
        "pred_noisy": noisy_preds,
        "confidence": np.round(test_clean_probs.max(axis=1), 4),
        "confidence_noisy": np.round(test_noisy_probs.max(axis=1), 4)
    })

    for idx, emotion in enumerate(EMOTIONS):
        out[f"prob_{emotion}"] = np.round(test_clean_probs[:, idx], 4)

    out.to_csv(RESULTS_DIR / "predictions_baseline_test.csv", index=False)

    # 6. Save Model Artifacts & Metadata
    joblib.dump(pipeline, MODELS_DIR / "baseline.joblib")
    joblib.dump(mlb, MODELS_DIR / "mlb.joblib")

    meta = {
        "labels": EMOTIONS,
        "aspect_seeds": ASPECT_SEEDS,
        "val_macro_f1": round(val_f1, 4),
        "val_accuracy": round(val_acc, 4),
        "test_clean_macro_f1": round(clean_f1, 4),
        "test_noisy_macro_f1": round(noisy_f1, 4),
        "seed": SEED,
        "model_type": "Subword TF-IDF + Calibrated LightGBM (5-Fold CV)"
    }
    (MODELS_DIR / "baseline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    table = update_model_comparison()
    print("\n" + table.to_string(index=False))
    print(f"\nPipeline training complete in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())