# News Category Classification — Implementation Plan

## Overview

- **Team:** 7 students, no specialisation preferences. Tasks distributed evenly; any member can pick up any task.
- **Total sprints:** 3
- **Sprint length:** 10 calendar days each (≤14-day rule satisfied)
- **Total duration:** ~30 days from kickoff
- **Target first milestone:** M0 by end of Sprint 1
- **Hard deadline:** ~2026-06-08 (one month from kickoff 2026-05-08); confirm exact date with course coordinator
- **Assumed start date:** 2026-05-09
- **Per-sprint capacity assumption:** 7 members × ~1.5 weeks × ~12h/week of effective work ≈ 100h. Conservative budget per sprint: **~80h of executable task work** (the remaining ~20h absorbs coordination, code review, learning curve, exam load).

## Release Milestones

Mapping from PRD §9 to sprints:

| Milestone | Delivered at end of |
|---|---|
| M0 — Vertical slice (data → cleaning → baseline → minimal Gradio) | Sprint 1 |
| M1 — Feature complete (all 7 models, full eval, full Gradio, LLM, retrieval) | Sprint 2 |
| M2 — Polished delivery (docs, slides, written report, rehearsed demo) | Sprint 3 |

---

## Sprint 1 — Foundation & Vertical Slice (10 days)

**Goal:** Stand up the repo, get the data flowing, and prove the full path from raw text to a category prediction shown in a Gradio UI — even if only one model exists and the LLM is mocked. Everything downstream reuses this foundation.

**Demo-able deliverable at sprint end:** A teammate runs the notebook on a fresh Colab, gets a Gradio link, types "Apple announces new iPhone with improved camera", and sees a predicted category come back from the trained Logistic Regression baseline.

**Estimated capacity used:** ~63h of ~80h budget.

### Tasks (in execution order)

- **S1-T1** [S] Create GitHub repo, configure `.gitignore`, set branch protection on `main`
  - Acceptance: repo exists; `main` requires PR + 1 review; `.gitignore` excludes `data/`, `models/`, `*.npy`, `*.joblib`, `kaggle.json`, `.env`, `__pycache__`, `.ipynb_checkpoints/`
  - Dependencies: none
  - Risk: low
- **S1-T2** [S] Pin baseline `requirements.txt` (Python 3.10, scikit-learn, pandas, nltk, transformers, torch, gradio, groq, joblib, matplotlib, seaborn, wordcloud, kaggle, pytest, ruff)
  - Acceptance: `pip install -r requirements.txt` succeeds in a fresh Colab; versions exact-pinned
  - Dependencies: S1-T1
  - Risk: low
- **S1-T3** [M] Scaffold `src/` package: empty modules (`data_loader.py`, `preprocessing.py`, `features.py`, `embeddings.py`, `train_ml.py`, `train_deep.py`, `evaluate.py`, `retrieval.py`, `llm.py`), `__init__.py`, top-level `app_gradio.py`, `notebooks/` folder, `data/`, `models/`, `reports/` placeholder dirs with `.gitkeep`
  - Acceptance: `python -c "import src"` works; folder layout matches PRD §F11 / spec
  - Dependencies: S1-T1
  - Risk: low
- **S1-T4** [M] CI workflow: GitHub Actions runs `ruff` lint + `pytest` smoke test on every PR
  - Acceptance: a trivial `tests/test_smoke.py` passes; lint failure blocks merge
  - Dependencies: S1-T3
  - Risk: low
- **S1-T5** [M] `data_loader.load_dataset()` — Kaggle download with local cache + JSON-Lines parse to DataFrame
  - Acceptance: returns DataFrame with `category, headline, short_description, date, authors, link`; second call uses cache; ≥ 200K rows; documented in module docstring
  - Dependencies: S1-T3
  - Risk: low (Kaggle API is well-known)
- **S1-T6** [L] `preprocessing.clean_text()` — full pipeline: lowercase → URL/HTML/punct/digit removal → NLTK tokenize → stopword removal → WordNet lemmatisation; plus a builder for `(word_count, char_count, punct_count)` numeric features; unit tests
  - Acceptance: 6 unit tests cover ordering, empty input, URL stripping, lemma forms, stopword removal, numeric feature correctness; all pass in CI
  - Dependencies: S1-T3
  - Risk: medium — order matters; bugs here invalidate every model
- **S1-T7** [M] EDA part 1 in `notebooks/01_eda.ipynb`: shape, dtypes, missing values, duplicates, basic stats, class-distribution bar chart with imbalance commentary
  - Acceptance: notebook runs top-to-bottom on Colab; charts render; commentary cells explain findings in plain English
  - Dependencies: S1-T5, S1-T6
  - Risk: low
- **S1-T8** [M] `features.build_classical_features()` — fit/transform `TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)`, build numeric features, sparse-stack
  - Acceptance: returns sparse matrix `(N, 50_003)`; vectorizer + scaler persisted to `models/`; round-trip test (fit → save → load → transform) passes
  - Dependencies: S1-T6
  - Risk: low
- **S1-T9** [L] `train_ml.train_baseline()` — Logistic Regression with `class_weight='balanced'`, stratified 80/20 split (`random_state=42`), simple `GridSearchCV` over `C ∈ {0.1, 1, 10}`; persist best model
  - Acceptance: `models/logreg_best.joblib` exists; macro-F1 ≥ 0.45 on test set (low bar — proves pipeline works); training reproducible across runs
  - Dependencies: S1-T8
  - Risk: medium — first ML training, may surface data issues
- **S1-T10** [M] `evaluate.evaluate_model()` skeleton — produce row of (accuracy, precision, recall, F1-macro, F1-weighted, ROC-AUC) and a confusion-matrix PNG; append row to `reports/model_comparison.csv`
  - Acceptance: function works for the LogReg baseline; CSV row exists; PNG saved to `reports/confusion/logreg.png`
  - Dependencies: S1-T9
  - Risk: low
- **S1-T11** [M] RoBERTa feasibility spike — load `roberta-base`, embed 100 sample sentences, mean-pool, save `(100, 768)` array; document peak memory and time
  - Acceptance: script runs end-to-end on Colab GPU; memory + timing logged in a markdown cell; cache mechanism prototype works
  - Dependencies: S1-T6
  - Risk: medium — Colab free-tier GPU constraints are the main known unknown
- **S1-T12** [L] Minimal Gradio app — two text inputs, one button, displays predicted category and confidence from the LogReg baseline; launches a public share link from Colab
  - Acceptance: live share link works from a fresh Colab session; app handles empty input gracefully
  - Dependencies: S1-T6, S1-T9
  - Risk: low
- **S1-T13** [S] Groq client smoke test — auth with `GROQ_API_KEY`, send a hello prompt, log the response
  - Acceptance: 1-line response received; failure path (missing key) gives a clear error message
  - Dependencies: S1-T3
  - Risk: low
- **S1-T14** [S] M0 sign-off — checklist run: repo green, CI green, notebook runs end-to-end on a teammate's fresh Colab, Gradio link prints, baseline prediction returns
  - Acceptance: filled-in checklist committed to `docs/progress.md`; M0 tag cut on `main`
  - Dependencies: all S1 tasks
  - Risk: low

### Sprint 1 exit criteria
- All task acceptance criteria checked ✅
- Demo script (Run All on a fresh Colab → click Predict → see a category) runs without intervention
- No blocking `TODO` left in `src/`
- `ruff` and `pytest` green
- `docs/progress.md` updated with Sprint 1 completion notes
- Tag `m0` pushed to `main`

---

## Sprint 2 — Feature Complete (10 days) — REVISED 2026-05-08

**Scope-change note:** ADRs 010, 011, 012 (logged at sprint-2 kickoff) consolidate categories from 42→27, replace the frozen-RoBERTa+LinearSVC bonus with the team coordinator's pre-trained fine-tuned RoBERTa, and put the model on Google Drive instead of cached embeddings. Old S2-T6 (full embedding extraction) and old S2-T7 (LinearSVC on embeddings) are retired in favour of S2-T2 (download + load) and S2-T8 (evaluate). The remaining classical models, EDA, LLM, retrieval, and Gradio tasks survive — they just train against the 27-class label space.

**Goal:** Five classical models trained and tuned on the 27-class merged dataset, plus the fine-tuned RoBERTa loaded from Drive and evaluated; full Gradio UI with LLM explanations + similarity retrieval; full EDA part 2.

**Demo-able deliverable at sprint end:** Gradio app with predicted category + confidence + Groq-generated explanation + top-3 similar articles. Best-of-7 model auto-selected from `model_comparison.csv`. 7 confusion-matrix PNGs saved.

**Estimated capacity used:** ~80h of ~80h budget.

### Tasks (in execution order)

- **S2-T1** [S] **Apply CATEGORY_MAP** in the data-loading + preprocessing cells: 42→27 labels per ADR-010, drop rows whose cleaned text is <4 words, drop exact duplicates. Re-fit `LabelEncoder` against the 27 labels. Persist the new `data/processed/cleaned.parquet`.
  - Acceptance: cleaned DataFrame has exactly 27 unique categories; row count ≈ 207K; `models/label_encoder.joblib` updated; sprint-1 LogReg cell re-runs against 27 classes (sanity check that the pipeline still works) and macro-F1 ≥ 0.60.
  - Dependencies: S1 complete
  - Risk: low
- **S2-T2** [M] **Download fine-tuned RoBERTa from Drive** (FILE_ID `19EIWqmmR4tbJrMyiqKYRT__s_d1n11rW`) into `models/best_model/`, load with `AutoTokenizer.from_pretrained()` + `AutoModelForSequenceClassification.from_pretrained()`. Cache mechanism: skip download if `models/best_model/config.json` exists.
  - Acceptance: model loads on a fresh Colab; `model.config.num_labels == 27`; predicting on a 5-row test sample produces sensible categories.
  - Dependencies: S2-T1
  - Risk: medium — depends on Drive availability and the FILE_ID staying public.
- **S2-T3** [L] Train + tune **Linear SVM** (`LinearSVC` wrapped in `CalibratedClassifierCV`) with grid over `C` on the 27-class data; persist; eval row added.
  - Acceptance: `models/linearsvc_best.joblib` saved; calibrated probabilities sum to 1; comparison row added; macro-F1 ≥ 0.60 (PRD §2 target).
  - Dependencies: S2-T1
  - Risk: low
- **S2-T4** [L] Train + tune **KNN** with `RandomizedSearchCV` over `n_neighbors ∈ {3,5,7,11,21}`, `weights ∈ {uniform, distance}`.
  - Acceptance: `models/knn_best.joblib` saved; comparison row added; note KNN's poor scaling in the comparison table.
  - Dependencies: S2-T1
  - Risk: medium — slow inference at this scale.
- **S2-T5** [L] Train + tune **Decision Tree** with grid over `max_depth`, `min_samples_leaf`; persist.
  - Acceptance: `models/decision_tree_best.joblib`; comparison row; tree depth + leaf count logged.
  - Dependencies: S2-T1
  - Risk: low
- **S2-T6** [L] Train + tune **Random Forest** with `RandomizedSearchCV` over `n_estimators ∈ {100,200,400}`, `max_depth`, `class_weight='balanced'`.
  - Acceptance: `models/rf_best.joblib`; comparison row; OOB score logged.
  - Dependencies: S2-T1
  - Risk: medium — long training; may need to subsample for tuning, refit best params on full.
- **S2-T7** [L] Train + tune **AdaBoost** with grid over `n_estimators` and `learning_rate`.
  - Acceptance: `models/adaboost_best.joblib`; comparison row.
  - Dependencies: S2-T1
  - Risk: low
- **S2-T8** [M] **Evaluate fine-tuned RoBERTa** (loaded in S2-T2) on the same held-out test set. Run inference in batches under `torch.no_grad()`, run the standard metric suite (`evaluate_model()` from S1-T10), append a row labelled `roberta_finetuned` to `reports/model_comparison.csv`.
  - Acceptance: comparison row added; **accuracy ≥ 0.70**, **macro-F1 ≥ 0.65** (PRD §2 criterion 4); confusion-matrix PNG `reports/confusion/roberta_finetuned.png` saved.
  - Dependencies: S2-T1, S2-T2
  - Risk: low — the model is pre-trained.
- **S2-T9** [M] Finalise the **comparison table** with all 7 rows (LogReg from sprint 1 + 5 sprint-2 classical + RoBERTa fine-tuned). Re-render the styled DataFrame in the notebook with the best model highlighted. Save 7 confusion-matrix PNGs in `reports/confusion/`.
  - Acceptance: `reports/model_comparison.csv` has 7 rows × 7 metric columns; 7 PNGs present; styled table in the notebook visually picks the best.
  - Dependencies: S2-T3 through S2-T8
  - Risk: low
- **S2-T10** [L] Full **EDA part 2** in `notebooks/01_eda.ipynb` against the 27-class data: numeric distributions (histograms), boxplots, correlation heatmap, token-length distribution, top-20 most frequent words, n-gram analysis (top 20 bi-grams + tri-grams), word cloud, outlier visualisation, skewness check, scaling-need check, noise/anomaly inspection.
  - Acceptance: every checkbox in spec §"Visualization & Analysis" rendered; each chart has a 1–2 line interpretation cell underneath.
  - Dependencies: S2-T1
  - Risk: low
- **S2-T11** [M] **Real Groq LLM integration**: prompt template from spec, Groq SDK call, in-process `lru_cache` keyed by `(sha1(text)[:16], label)`, network/error handling.
  - Acceptance: `explain(text, label, confidence)` returns a clean 2–3 sentence string for 9/10 hand-rated samples; errors return a friendly placeholder; cache hit rate verified.
  - Dependencies: S1-T13
  - Risk: medium — rate-limit handling, prompt quality.
- **S2-T12** [M] **Similarity retriever**: at startup, run the fine-tuned model's encoder over the (cleaned) training set, mean-pool last hidden states, cache `(N, 768)` to `data/embeddings/roberta_finetuned.npy` + a parallel `index.parquet` of `(headline, category)`. `top_k(query_text, k=3)` cosine-ranks against the cache.
  - Acceptance: cache file produced (~600 MB); `top_k` returns `[(headline, category, score), ...]` in < 200 ms after cache load.
  - Dependencies: S2-T2
  - Risk: medium — embedding pass over 207K rows on Colab T4 takes ~30-45 min on first run; cached after that.
- **S2-T13** [L] **Gradio app — full version**: select best model from `model_comparison.csv` (will pick `roberta_finetuned`), show category + confidence (%) + LLM explanation + top-3 similar articles; loading spinner; graceful error UI.
  - Acceptance: end-to-end inference on warm runtime < 5 s including LLM; on Groq error the UI degrades gracefully (placeholder explanation, rest still renders).
  - Dependencies: S2-T9, S2-T11, S2-T12
  - Risk: medium
- **S2-T14** [M] **Integration test** — full notebook re-runs on a fresh Colab from a teammate's account: bootstrap → category map → Drive download → classical training → RoBERTa eval → Gradio launch.
  - Acceptance: time-to-Gradio-link recorded; under 90 minutes cold-start; under 10 minutes warm.
  - Dependencies: S2-T13
  - Risk: medium
- **S2-T15** [S] M1 sign-off — feature checklist verified against PRD §5 MVP; tag `m1` cut on `main`.
  - Acceptance: every MVP feature F1–F11 has its acceptance criteria checked; sign-off committed to `docs/progress.md`.
  - Dependencies: all S2 tasks
  - Risk: low

### Sprint 2 exit criteria
- All MVP features (F1–F11) verified against their PRD acceptance criteria ✅
- 7 trained models in `models/` (5 new sprint-2 classical + S1 LogReg + RoBERTa fine-tuned downloaded), 7 rows in the comparison table, 7 confusion-matrix PNGs
- 27-class label space throughout; sprint-1 logreg row in the comparison table re-evaluated on 27 classes
- Notebook runs end-to-end on a fresh Colab without manual fixes (with `kaggle.json` + `GROQ_API_KEY` provided)
- Gradio app shows all four output components live
- `docs/progress.md` updated; tag `m1` pushed

---

## Sprint 3 — Polish & Delivery (10 days)

**Goal:** Convert a working system into a graded submission. Documentation, slides, written report, rehearsals, bug fixes.

**Demo-able deliverable at sprint end:** A 10-minute live demo run twice without surprises; ZIP + GitHub link ready to submit; every required deliverable in the spec exists.

**Estimated capacity used:** ~57h of ~80h budget — keeps a real buffer for the inevitable last-week fixes.

### Tasks (in execution order)

- **S3-T1** [L] **Notebook polish** in `notebooks/00_main.ipynb`: clear section headers (Setup → Data → EDA → Features → Models → Evaluation → Demo), minimal narrative, final summary cell that prints dataset size, split sizes, all model metrics, total wall-time
  - Acceptance: notebook is code-focused (per spec); reading the section headers tells the full story; `Run All` produces no warnings
  - Dependencies: M1
  - Risk: low
- **S3-T2** [M] **README.md**: project overview, setup steps, API-key instructions (Kaggle + Groq), how to run, file map, team credits (no mention of any AI assistant)
  - Acceptance: README walks a stranger from clone → working Gradio link in under 15 minutes
  - Dependencies: M1
  - Risk: low
- **S3-T3** [S] Freeze `requirements.txt` to the exact versions used in the final Colab session
  - Acceptance: `pip freeze` output sanitised (only direct deps + transitive) committed
  - Dependencies: M1
  - Risk: low
- **S3-T4** [L] **Slide deck** (no code on slides): problem statement, dataset overview, methodology, EDA highlights, model comparison, demo screenshots, lessons learned, future work
  - Acceptance: ≤ 15 slides; every slide is readable from the back of the room; diagrams from `architecture.md` reused
  - Dependencies: M1
  - Risk: low
- **S3-T5** [L] **Written report** — plain-language explanation of the concept aimed at a non-ML audience: what the problem is, how text is turned into numbers, what each model does in one sentence, why we chose what we chose, how the LLM helps, limits of the system
  - Acceptance: ~3–5 pages; no formulas; no code; passes a "would my parents understand?" read-aloud test
  - Dependencies: M1
  - Risk: low
- **S3-T6** [M] **Pick the final best model** based on macro-F1 + qualitative review of confusion matrices; document the choice in the report and pin it as the default in `app_gradio.py`
  - Acceptance: written justification cited in both the report and the notebook; `BEST_MODEL` constant points to the chosen file
  - Dependencies: S2-T8
  - Risk: low
- **S3-T7** [M] **Confusion-matrix cleanup**: re-render the chosen best model's confusion matrix at higher resolution with axis labels readable for the slides; write a short paragraph on the worst-confused class pairs
  - Acceptance: PNG embedded in slides + report; commentary identifies at least 3 confused class pairs
  - Dependencies: S3-T6
  - Risk: low
- **S3-T8** [L] **Demo rehearsal #1** — full 10-minute walk-through with timing; record screen
  - Acceptance: completes in 9–11 minutes; rehearsal notes list every fix needed
  - Dependencies: S3-T1, S3-T4
  - Risk: medium — first time integrating slides + live demo
- **S3-T9** [M] **Bug-fix sweep** based on rehearsal #1 findings (UI quirks, slow paths, slide typos, broken links)
  - Acceptance: every rehearsal-note item closed or explicitly deferred with rationale
  - Dependencies: S3-T8
  - Risk: medium
- **S3-T10** [L] **Demo rehearsal #2** — final dry run with the full team in the room
  - Acceptance: clean 10-minute run; everyone knows their hand-off; backup plan ready (pre-recorded screencast in case the live link fails)
  - Dependencies: S3-T9
  - Risk: low
- **S3-T11** [M] **Submission package**: tag `v1.0` on `main`; export ZIP of the final state; verify GitHub link is public-readable; double-check all artefacts (notebook, app, slides, report, README, requirements.txt) are committed
  - Acceptance: a teammate clones the public repo on a different machine, runs the README steps, and gets a working Gradio link
  - Dependencies: S3-T10
  - Risk: medium — last-minute issues are the highest-cost ones
- **S3-T12** [S] Buffer / unforeseen fixes — keep the slot open
  - Acceptance: used or explicitly closed unused
  - Dependencies: none
  - Risk: low

### Sprint 3 exit criteria
- All 4 deliverables present: structured Jupyter notebook ✅, Gradio app ✅, slides ✅, written report ✅, plus README + requirements.txt
- Tag `v1.0` pushed
- Final demo rehearsal completed without surprises
- `docs/progress.md` reflects M2 completion

---

## Post-MVP Roadmap

Not part of this delivery, but for the team's longer-term thinking:

- Permanent hosting on Hugging Face Spaces (currently Colab-only).
- Multilingual support (currently English-only).
- Top-k display in the UI (PRD F13).
- Calibration sweep across all probabilistic models (PRD F12).
- Fine-tune RoBERTa end-to-end and compare against frozen-embeddings baseline.
- Compare prompt-only LLM classification vs. the trained classifier.

---

## Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Sprint |
|---|---|---|---|---|---|
| R1 | Colab free-tier GPU session limits cut RoBERTa embedding extraction off mid-run | Med | High | Aggressive batching + cache; subsample fallback to stratified 100K if 200K fails. Document either way. | S2 |
| R2 | 7-person merge conflicts on the notebook | Med | Med | Logic lives in `src/*.py` (small text-mergeable files); notebook is thin and section-owned by named members. | S1+ |
| R3 | Groq free-tier rate limits during the live demo | Low | High | LLM cache warmed before demo; UI degrades gracefully when LLM fails so the rest of the demo continues. | S2 / S3 |
| R4 | Hyper-parameter tuning takes too long with 200K rows × 6 models | Med | Med | `RandomizedSearchCV` with small param grids; tune on a 50K stratified subsample, refit best params on the full 200K. | S2 |
| R5 | Macro-F1 below the PRD target due to extreme class imbalance | Med | Med | Report both macro and weighted F1; transparently discuss the limitation in the written report; do not over-claim in slides. | S2 / S3 |
| R6 | Team availability uneven (exams, illness) over the month | High | Med | Tasks sized small (≤ L); no critical-path single-owner task is larger than 1.5 days; sprint capacity has a 20% buffer. | All |
| R7 | Last-week submission crunch | Med | High | M2 lands at end of Sprint 3 with deliberate buffer (~25h unused budget); rehearsal #2 occurs at least 2 days before deadline. | S3 |
| R8 | API keys re-leaked or expired before demo | Low | High | Both keys exposed during planning are already rotated; secrets live only in `kaggle.json` + `GROQ_API_KEY` env var; demo session re-auths fresh. | All |
| R9 | Notebook works on author's Colab but not on grader's | Low | High | Sprint 2 includes integration test on a teammate's fresh Colab; Sprint 3 includes a clean clone-and-run by a teammate on a different machine. | S2 / S3 |

---

## Assumptions

- Course deadline is approximately 30 days from kickoff (2026-06-08). If the actual deadline is sooner, drop F12/F13 and trim Sprint 3 polish.
- All 7 members can dedicate ~12h/week to the project (~3h/day weekday equivalent on average).
- Colab free tier is the baseline runtime. Anyone with Colab Pro is a bonus, not a requirement.
- Both Kaggle and Groq remain available throughout the project window.
- No additional bonus work is layered on past what's already agreed; new asks become Post-MVP.
