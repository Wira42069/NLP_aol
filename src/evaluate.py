"""Evaluation and Error Analysis (Phase 4b).

Generates metrics comparing model performance on Clean vs. Noisy (Typos) text.

Run as a standalone script to generate errors for manual review:
    python src/evaluate.py
    python src/evaluate.py --report-examples

Outputs
-------
results/model_comparison.csv      Clean vs. Noisy Test Set metrics
results/confusion_clean.csv       Confusion matrix (Clean)
results/confusion_noisy.csv       Confusion matrix (Noisy)
results/errors.csv                Sample of misclassified reviews for human annotation
results/report_examples.md        Formatted markdown of annotated errors (if requested)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import EMOTIONS  # noqa: E402

RESULTS_DIR = ROOT / "results"


def update_model_comparison() -> pd.DataFrame:
    """Calculates metrics for Clean vs Noisy test sets and saves the comparison table."""
    pred_path = RESULTS_DIR / "predictions_baseline_test.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}. Run train_baseline.py first.")

    df = pd.read_csv(pred_path)
    metrics = []

    configs = [
        ("Clean Test Set", "pred", "confidence"),
        ("Noisy Test Set (Typos)", "pred_noisy", "confidence_noisy"),
    ]

    for condition, pred_col, conf_col in configs:
        if pred_col not in df.columns:
            continue
        
        f1 = f1_score(df["true"], df[pred_col], average="macro")
        acc = accuracy_score(df["true"], df[pred_col])
        avg_conf = df[conf_col].mean()
        
        metrics.append({
            "Evaluation Target": condition,
            "Macro-F1": round(f1, 4),
            "Accuracy": round(acc, 4),
            "Avg Confidence": round(avg_conf, 4)
        })

    comp_df = pd.DataFrame(metrics)
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    return comp_df


def save_confusion_matrices(df: pd.DataFrame) -> None:
    """Saves standard confusion matrices for both clean and noisy predictions."""
    for condition, pred_col in [("clean", "pred"), ("noisy", "pred_noisy")]:
        if pred_col not in df.columns:
            continue
            
        cm = confusion_matrix(df["true"], df[pred_col], labels=EMOTIONS)
        cm_df = pd.DataFrame(cm, index=[f"True_{e}" for e in EMOTIONS], columns=[f"Pred_{e}" for e in EMOTIONS])
        cm_df.to_csv(RESULTS_DIR / f"confusion_{condition}.csv")


def generate_error_samples(df: pd.DataFrame, n_samples: int = 40) -> None:
    """Samples misclassifications from the clean test set for manual categorization."""
    errors = df[df["true"] != df["pred"]].copy()
    if len(errors) == 0:
        print("No errors found in the test set.")
        return

    sampled = errors.sample(n=min(n_samples, len(errors)), random_state=42)
    
    out = pd.DataFrame({
        "id": sampled["id"],
        "text": sampled["text"],
        "true_emotion": sampled["true"],
        "predicted_emotion": sampled["pred"],
        "confidence": sampled["confidence"],
        "error_category": "",  # To be filled by hand
        "human_notes": ""      # To be filled by hand
    })
    
    out.to_csv(RESULTS_DIR / "errors.csv", index=False)
    print(f"Exported {len(out)} misclassified samples to results/errors.csv for manual review.")


def format_report_examples() -> None:
    """Formats manually annotated errors into a Markdown file for the final report."""
    errors_path = RESULTS_DIR / "errors.csv"
    if not errors_path.exists():
        print("results/errors.csv not found. Run evaluate.py first.")
        return

    df = pd.read_csv(errors_path)
    annotated = df[df["error_category"].notna() & (df["error_category"] != "")]
    
    if len(annotated) == 0:
        print("No manual annotations found in results/errors.csv. Please fill the 'error_category' column.")
        return

    lines = ["## Error Analysis Examples\n"]
    for cat, group in annotated.groupby("error_category"):
        lines.append(f"### Category: {cat}\n")
        for _, row in group.head(3).iterrows():
            lines.append(f"- **Review:** \"{row['text']}\"")
            lines.append(f"  - **True:** {row['true_emotion']} | **Predicted:** {row['predicted_emotion']} ({row['confidence']:.0%})")
            if pd.notna(row["human_notes"]):
                lines.append(f"  - **Notes:** {row['human_notes']}")
            lines.append("")

    out_path = RESULTS_DIR / "report_examples.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-examples", action="store_true", help="Generate Markdown from annotated errors.csv")
    args = parser.parse_args(argv)

    if args.report_examples:
        format_report_examples()
        return 0

    RESULTS_DIR.mkdir(exist_ok=True)
    table = update_model_comparison()
    print("\nModel Evaluation Summary:\n")
    print(table.to_string(index=False))

    df = pd.read_csv(RESULTS_DIR / "predictions_baseline_test.csv")
    save_confusion_matrices(df)
    generate_error_samples(df)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())