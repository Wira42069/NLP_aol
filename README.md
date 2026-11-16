  


# Tokopedia Review Emotion Dashboard for Sellers (CPU-Optimized)

  

Classifies Indonesian Tokopedia product reviews into five emotions (**happy, love, sadness, fear, anger**) and tags which e-commerce aspect each review is about (shipping, quality, packaging, pricing, seller service) using semantic vector proximity.

  

Designed for rapid deployment on standard laptops, this project enables small sellers (UMKM) to see *what* customers talk about and *how they feel* about it without requiring GPU infrastructure.

  

- **Dataset:** PRDECT-ID (~5,400 Indonesian Tokopedia reviews)

- **Emotion Engine:** Subword TF-IDF + Calibrated LightGBM (5-Fold Cross-Validation)

- **Aspect Engine:** Unsupervised FastText Cosine Similarity

- **App:** Streamlit, 2 pages (Single Review Inspector, Batch Seller Dashboard)

  

> **Status of this repository:** The pipeline has been completely rewritten from the original IndoBERT spec to run end-to-end on a local CPU. The dashboard now outputs calibrated probabilities and utilizes a synthetic robustness test to measure real-world resilience against typos.

  

---

  

## 1. Install

  

This architecture is completely CPU-native. No PyTorch, Transformers, or Colab GPUs are required.

  

```bash

python -m venv .venv

source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

  


  

## 2. Get the data (manual download, not committed to git)

  

* **PRDECT-ID:** Download the CSV from its Mendeley Data page ("Product Reviews Dataset for Emotions Classification Tasks - Indonesian (PRDECT-ID)") and put it in `data/raw/`. Cite the dataset paper in the report. Expected columns: `Customer Review`, `Emotion`, `Sentiment` (others are ignored).

* *(Note: The external slang dictionary requirement has been removed. The pipeline now natively handles slang via Character N-Grams and FastText subwords).*

  

## 3. Data Prep & Robustness Split (Phase 0-1)

  

```bash

python src/data_loader.py

  



  

**What this does:**

  

1. Uses `datasketch` MinHash to drop near-duplicate reviews (e.g., copy-pasted seller templates).

2. Creates a clean 10% holdout test set and builds 5 Stratified CV folds from the remaining 90%.

3. Injects synthetic Indonesian marketplace typos into a `text_noisy` column in the test set to evaluate real-world robustness.

  

## 4. Train the Pipeline (Phases 2-3)

  

```bash

python src/train_baseline.py

  



  

**What this does:**

  

1. Runs 5-Fold Cross-Validation using a combination of Word (1-2 gram) and Character (3-5 gram) TF-IDF features to natively handle slang and typos.

2. Trains Calibrated LightGBM One-Vs-Rest classifiers and logs metrics.

3. Trains a FastText embedding model on the review corpus for semantic aspect mapping.

4. Generates predictions for both the *Clean* and *Noisy* test sets, saving artifacts to `models/`.

  

## 5. Evaluate + Error Analysis (Phase 4b)

  

```bash

python src/evaluate.py

  



  

This prints the comparison between Clean and Noisy (Robustness) test metrics, saves confusion matrices, and generates `results/errors.csv` containing sampled misclassifications.

  

Open `results/errors.csv` and fill the `error_category` column by hand with exactly one of `Sarcasm/Irony`, `Ambiguous/Multi-Emotion`, `Slang/Spelling Outliers`, or `Annotation Noise`. Then run:

  

```bash

python src/evaluate.py --report-examples      # Extracts your annotations to markdown

python src/make_report_tables.py              # Generates docs/report_tables.md

  



  

## 6. Run the App

  

```bash

streamlit run app/app.py

  


  

Sidebar: choose between the Single Review Inspector and the Batch Seller Dashboard. The batch page accepts any CSV with a review text column; use `data/demo_reviews.csv` (50 hand-written reviews, *not* from PRDECT-ID) for the demo video.

  

## 7. Remaining human work

  

| Item | Where | Owner |
| --- | --- | --- |
| Peer/seller feedback sessions (3-5 people, 15 min) | `docs/user_feedback.md` | feedback lead |
| Keep logging every AI-assisted task | `docs/ai_usage_log.md` | everyone |
| Write the report (10-15 pages) | `docs/report_draft.md` -> final document | report editor |
| Record the demo video (< 5 min) | `docs/demo_script.md` | video lead |

  

## Rubric map

  

| Rubric item | Where it lives |
| --- | --- |
| Runnable Streamlit app, single + batch | `app/app.py` |
| Pipeline Architecture (TF-IDF + LightGBM + FastText) | `src/train_baseline.py`, `src/inference.py` |
| Predictions, calibrated confidence, aspect breakdown | both pages, `src/inference.py` |
| Robustness (Clean vs Noisy) metrics table | `results/model_comparison.csv` -> `docs/report_tables.md` |
| 4-category error analysis | `results/errors.csv`, `results/report_examples.md` |
| AI usage log (mandatory) | `docs/ai_usage_log.md` |
| Demo video script | `docs/demo_script.md` |

  

## Project rules & Architecture Decisions

  

1. **Fuzzy Deduplication over Exact Matching:** Split occurs before preprocessing. We use `datasketch` MinHash to prevent data leakage from near-duplicate seller templates. `tests/` checks the split is disjoint and fixed-seed.

2. **No Stemming or Static Dictionaries:** Sastrawi stemming and static slang dictionaries are removed. We rely entirely on Subword Character TF-IDF (for emotion) and FastText Subword vectors (for aspects) to natively capture Indonesian spelling variations, typos, and slang.

3. **Calibrated Probabilities:** The pipeline uses Platt Scaling (`CalibratedClassifierCV`) so that dashboard confidence percentages represent true statistical likelihoods.

4. **All metrics from code:** Every number in tables/charts/report comes from a file in `results/` (use `make_report_tables.py`).

5. **AI Logging:** Log AI use continuously in `docs/ai_usage_log.md`.

  

Run the unit tests any time with `pip install pytest && pytest`.

  
