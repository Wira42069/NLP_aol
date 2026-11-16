# AI Usage Log (mandatory - 10 % of the rubric)

Log **every** AI-assisted task: code generation, debugging, report editing, brainstorming, translation. One row per
task, written the same day. "Verification" must say what *you* did to check the output (ran it, compared with the
source, read it against the data) - not just "looked fine". Add rows at the bottom; never delete rows.

| Date | Tool | Task / Prompt | Verification Method | Outcome / Adjustments |
| --- | --- | --- | --- | --- |
| 2026-10-06 | Claude (Sonnet 5.5) | Prompt: build the whole project from the "Streamlined Handoff Brief" - data loader and split, preprocessing, TF-IDF baseline with ablation, IndoBERT Colab script, evaluation and error-analysis tooling, insights module, Streamlit app, docs templates, demo script | Ran every script end to end on **synthetic** reviews in a scratch copy; 8 unit tests; headless tests of both app pages with both models; IndoBERT script dry-run on a tiny randomly initialised BERT (the real IndoBERT weights could not be downloaded in that environment) | Testing found and fixed two bugs (a crash in the "every aspect it mentions" matrix mode; `warmup_ratio` removed in transformers 5). **Not yet run on the real PRDECT-ID data or the real IndoBERT weights - the team must re-run and re-verify (`<name>`, `<date>`).** |
| | | | | |

## What the team still has to do for this log
- Add a row each time a teammate uses an AI tool (including for the report text and the video script).
- When AI text goes into the report, note which sections and how you checked facts and numbers against `results/`.
- Final report Section 10 = this table + a 3-4 sentence summary of how AI was used and verified.
