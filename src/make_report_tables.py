"""Turn the files in results/ into ready-to-paste Markdown tables (core rule 2: no hand-typed numbers).

    python src/make_report_tables.py        # writes docs/report_tables.md

Copy the tables from docs/report_tables.md into the report. Re-run it whenever results change.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "report_tables.md"


def md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = [float_fmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def section(title: str, body: str, source: str) -> str:
    return f"## {title}\n\n_Source: `{source}`_\n\n{body}\n"


def main() -> int:
    parts = ["# Report tables (auto-generated - do not edit by hand)\n"]

    eda_path = RESULTS / "eda_summary.json"
    if eda_path.exists():
        eda = json.loads(eda_path.read_text(encoding="utf-8"))
        raw = eda["raw_file_checks"]
        parts.append(section(
            "Dataset overview",
            md_table(pd.DataFrame([
                ["Rows in raw file", raw["n_rows_raw"]],
                ["Exact duplicate texts dropped", raw["n_exact_duplicates_dropped"]],
                ["Rows used", raw["n_rows_final"]],
                ["Missing text / emotion / sentiment", f"{raw['missing_text']} / {raw['missing_emotion']} / {raw['missing_sentiment']}"],
                ["Imbalance ratio (largest / smallest class)", eda["imbalance_ratio_max_to_min"]],
            ], columns=["Item", "Value"])),
            "results/eda_summary.json"))
        dist = pd.DataFrame({
            "emotion": list(eda["label_distribution_all"]["counts"]),
            "count": list(eda["label_distribution_all"]["counts"].values()),
            "percent": list(eda["label_distribution_all"]["percent"].values()),
        })
        parts.append(section("Label distribution", md_table(dist, "{:.2f}"), "results/eda_summary.json"))
        L = eda["length_words"]
        parts.append(section("Review length (words)", md_table(pd.DataFrame([L]), "{:.2f}"), "results/eda_summary.json"))

    for fname, title in [
        ("ablation.csv", "Preprocessing ablation (best validation Macro-F1 per condition)"),
        ("model_comparison.csv", "Model comparison (TEST set)"),
        ("per_class_baseline.csv", "Per-class test metrics - baseline"),
        ("per_class_indobert.csv", "Per-class test metrics - IndoBERT"),
    ]:
        p = RESULTS / fname
        if p.exists():
            df = pd.read_csv(p)
            if fname.startswith("per_class"):
                df = df.rename(columns={df.columns[0]: "class"})
            parts.append(section(title, md_table(df), f"results/{fname}"))

    cc = RESULTS / "error_category_counts.json"
    if cc.exists():
        counts = json.loads(cc.read_text(encoding="utf-8"))
        parts.append(section("Error categories (hand-labelled sample)",
                             md_table(pd.DataFrame(list(counts.items()), columns=["Category", "Errors"])),
                             "results/error_category_counts.json"))

    if len(parts) == 1:
        print("results/ is empty - run the pipeline first.")
        return 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(parts) - 1} tables)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
