# Demo Video Script (target 4:30, hard limit 5:00)

Rubric: **problem statement -> architecture -> app walkthrough -> results**. Record the screen, one narrator, one take
per section, then trim. Fill the `[...]` slots with values read from `results/` right before recording.

## Before recording (checklist)
- [ ] `streamlit run app/app.py` works with **both** models available in the sidebar
- [ ] Browser zoom 110-125 %, notifications off, only the app and one slide window open
- [ ] `data/demo_reviews.csv` ready; one single review of each emotion saved in a notepad to paste
- [ ] `results/model_comparison.csv`, one confusion-matrix PNG and `results/report_examples.md` open in tabs
- [ ] Rehearse once with a timer; speak slowly, about 130 words per minute

## Timeline

| Time | Screen | Narration (adapt in your own words) |
|---|---|---|
| 0:00-0:30 | Title slide, then a few raw Tokopedia-style reviews | "Small Tokopedia sellers get hundreds of reviews and a star rating tells them little. We built a dashboard that tells a seller how customers *feel* - happy, love, sadness, fear or anger - and *what* they talk about." |
| 0:30-1:15 | Architecture slide / diagram | "We use PRDECT-ID, about 5,400 Indonesian reviews. We split 80/10/10 before any preprocessing. A TF-IDF + Logistic Regression baseline is compared with a fine-tuned IndoBERT trained on Colab. A keyword module tags five aspects: shipping, quality, packaging, pricing and seller service. Everything is served in a Streamlit app." |
| 1:15-2:15 | **Single Review Inspector** | Paste a review -> point at the predicted emotion, confidence and probability bars. "Here is the primary aspect." Switch the sidebar from Baseline to IndoBERT with the same review and say whether they agree. |
| 2:15-3:30 | **Batch Seller Dashboard** | "Load the demo reviews" -> Run analysis -> emotion breakdown chart -> aspect x emotion matrix ("most anger sits in [aspect]") -> "Where to look first" -> filter the table -> click download CSV. |
| 3:30-4:30 | `model_comparison.csv`, confusion matrix, 1-2 error examples | "On the test set the baseline reaches Macro-F1 [..] and IndoBERT [..]. The most confused pair is [..]. Typical errors are [category] like [short example]." Mention one honest limitation. |
| 4:30-4:50 | Closing slide | "Next steps: aspect-level emotion and calibration. Thanks for watching." |

## Tips
- Never read a number that is not in `results/`; read it off the screen.
- Cut dead time (model loading, upload) in the editor; keep the final export under 5:00.
- Export 1080p MP4; test the file on another computer before submitting.
