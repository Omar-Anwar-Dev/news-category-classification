# News Category Classification — Product Requirements Document

## 1. Overview

This project builds an end-to-end **multi-class text classification system** that assigns one of **42 news categories** to a news article using its **headline** and **short description**. It is delivered as a runnable Google Colab notebook plus a Gradio demo and is built against the spec in `Docs/news_category_classification.pdf`.

The system is the deliverable for an academic Data Science course (Level 4) and is built by a **7-person student team** over **~4 weeks**. It is not a commercial product; "users" here means the demo audience (course staff and peers) plus the team itself.

The project intentionally goes beyond the minimum spec to also include the two declared bonuses — **RoBERTa contextual embeddings** for an alternative classifier, and a **Groq-hosted LLM** that explains why a given article was placed in its predicted category.

---

## 2. Goals & Success Metrics

### Primary goal
Deliver a complete, reproducible news-category classifier that satisfies every checkbox in the assignment spec, demoable live through a Gradio public link, and accompanied by clean documentation.

### Measurable success criteria

| # | Criterion | Target |
|---|---|---|
| 1 | All 6 classical models trained, tuned, and reported in a comparison table | 100% present |
| 2 | Best classical model **macro-F1** on held-out test set | ≥ 0.55 |
| 3 | Best classical model **weighted-F1** on held-out test set | ≥ 0.65 |
| 4 | RoBERTa + Linear-SVM model trained and reported | present, macro-F1 ≥ 0.60 |
| 5 | Gradio demo launches end-to-end on a fresh Colab runtime | < 10 min from `Run All` |
| 6 | LLM explanation returns a coherent 2-3 sentence rationale | ≥ 9 / 10 hand-rated samples |
| 7 | Notebook re-runs cleanly with no manual intervention beyond providing API keys | reproducible |
| 8 | All 4 deliverables exist: notebook, Gradio app, slides, written report, README | 100% delivered |

Macro-F1 thresholds reflect that 42-class news classification with TF-IDF typically reaches 0.55–0.65 macro-F1 in the literature; these are realistic stretch targets, not arbitrary.

### Non-goals

- **Not** a real-time news scraper or aggregator.
- **Not** deployed to a permanent public URL (Gradio share link is demo-only).
- **Not** multilingual — English only.
- **Not** a research paper — we are implementing known techniques cleanly, not pushing SOTA.
- **No** authentication, no user accounts, no persistence of user queries.
- **No** fine-tuning of RoBERTa (we use frozen embeddings + downstream SVM).

---

## 3. User Personas

### P1 — Course Grader / Instructor
- **Context:** Reviews 10+ submissions; needs to verify against a checklist quickly.
- **Cares about:** Completeness vs spec, code quality, reproducibility, working live demo, presentation clarity.
- **Frustrations:** Notebooks that don't run; missing required artefacts; over-claiming results.

### P2 — Demo Audience (peers, instructor during live session)
- **Context:** Watches a 10-minute demo of the Gradio app.
- **Cares about:** That the prediction looks reasonable, that explanations make sense, that the UI doesn't break.
- **Frustrations:** Slow inference; obviously wrong predictions on common topics; LLM gibberish.

### P3 — Team Member (developer)
- **Context:** One of 7 students; takes ownership of a slice of the pipeline.
- **Cares about:** Modular code so their part doesn't break others; clear interfaces; ability to run their slice in isolation.
- **Frustrations:** Notebook merge conflicts; tightly coupled functions; missing dependencies.

---

## 4. User Stories

Grouped by persona.

### Grader stories
- As a grader, I want to **open the notebook and run-all in Colab** so I can verify reproducibility without configuring an environment.
- As a grader, I want a **clear comparison table of all models** so I can see which performed best and by how much.
- As a grader, I want **all required EDA charts** in the notebook so I can confirm the data-understanding section is complete.
- As a grader, I want a **README with setup steps and an API-key checklist** so I know what I need to provide.

### Demo audience stories
- As a demo user, I want to **paste a headline + description and click predict** so I see a category, confidence, and a short explanation.
- As a demo user, I want to **see similar articles** so I can sanity-check the prediction.
- As a demo user, I want the prediction to come back **in under 5 seconds** so the demo doesn't drag.

### Team member stories
- As a team member, I want **plain `.py` modules under `src/`** so I can edit my piece in a regular IDE and import it from the notebook.
- As a team member, I want a **shared `requirements.txt`** so everyone has the same versions.
- As a team member, I want **a feature branch per task** so I don't conflict with teammates.

Each MVP feature in §5 maps acceptance criteria back to these stories.

---

## 5. Features

### MVP (in scope for this delivery — covers spec checkboxes + agreed bonuses)

#### F1 — Kaggle data ingestion
Download the News Category Dataset (~200K articles) from Kaggle using the official API. Parse JSON-Lines into a DataFrame.
- **Stories served:** Grader (reproducibility), Team member (clean data interface)
- **Acceptance criteria:**
  - `load_dataset()` returns a DataFrame with at least `category`, `headline`, `short_description`, `date`.
  - Re-running the function uses a local cache; does not re-download.
  - README documents the `kaggle.json` requirement.
- **Out of scope:** scraping arbitrary news sites; multiple datasets.

#### F2 — Text preprocessing pipeline
Lowercase → strip URLs/HTML/punctuation/digits → tokenize (NLTK) → remove stopwords → lemmatize (WordNet).
- **Stories served:** Grader (spec compliance), Team member (single source of truth for cleaning)
- **Acceptance criteria:**
  - One function `clean_text(s: str) -> str` covers all steps in the spec's stated order.
  - Same function used at training and inference time (proven by a unit test).
  - Cleaned dataset persisted to `data/processed/cleaned.parquet`.
- **Out of scope:** spelling correction, named-entity removal, language detection.

#### F3 — Comprehensive EDA
All charts listed under the spec's "Visualization & Analysis" section.
- **Stories served:** Grader (completeness)
- **Acceptance criteria:**
  - Dataset shape, dtypes, missing values, duplicates, basic stats — all printed.
  - Class-distribution bar chart with imbalance commentary.
  - Numeric feature distributions, boxplots, correlation heatmap.
  - Token-length distribution; top-20 most frequent words; n-gram analysis (top bi-grams + tri-grams); word cloud.
  - Outlier visualisation, skewness check, scaling-need check, noise/anomaly notes.
- **Out of scope:** interactive dashboards.

#### F4 — Classical feature engineering
TF-IDF (uni+bi-gram, capped vocabulary, sublinear-TF) + word/character/punctuation count features.
- **Stories served:** Grader (spec compliance)
- **Acceptance criteria:**
  - `TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True)` is fit on train only.
  - Numeric features computed from cleaned text and stacked sparsely with TF-IDF.
  - Vectorizer + scaler persisted to `models/`.
- **Out of scope:** word2vec, GloVe.

#### F5 — Six classical models with tuning
Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, AdaBoost — each trained with cross-validation and hyper-parameter tuning.
- **Stories served:** Grader (spec compliance)
- **Acceptance criteria:**
  - Each model has a documented param grid and best-params printed.
  - Each model is persisted to `models/<name>_best.joblib`.
  - `class_weight='balanced'` (or equivalent) applied where supported.
  - Stratified 80/20 train/test split with `random_state=42`.
- **Out of scope:** Neural classical models, AutoML.

#### F6 — Evaluation suite
Accuracy, Precision, Recall, F1 (macro and weighted), ROC-AUC (one-vs-rest), per-model confusion matrix, model-comparison table.
- **Stories served:** Grader (spec compliance)
- **Acceptance criteria:**
  - `reports/model_comparison.csv` with one row per model, all metrics columns.
  - One confusion-matrix PNG per model in `reports/confusion/`.
  - Comparison table also rendered in the notebook as a styled DataFrame.
- **Out of scope:** statistical significance testing.

#### F7 — RoBERTa embeddings + Linear SVM (Bonus)
Sentence → `roberta-base` → mean-pooled 768-d vector → LinearSVC.
- **Stories served:** Grader (bonus completeness)
- **Acceptance criteria:**
  - Embeddings cached to `data/embeddings/roberta.npy`; second run loads from cache.
  - LinearSVC scored using the same evaluation suite as F6 and added to the comparison table.
- **Out of scope:** RoBERTa fine-tuning; alternative pooling experiments.

#### F8 — LLM explanations (Bonus)
Groq Llama 3.3 70B generates a 2-3 sentence rationale for a predicted category, using the spec's prompt template.
- **Stories served:** Demo audience (sanity check), Grader (bonus completeness)
- **Acceptance criteria:**
  - `explain(text, label, confidence) -> str` returns a coherent paragraph in ≤ 3 sentences.
  - Failures (rate-limit, network) caught and replaced with a placeholder string; UI never crashes.
  - Results cached in-process by `(sha1(text), label)`.
- **Out of scope:** using the LLM to classify; multi-turn conversation.

#### F9 — Similar-article retrieval
Cosine similarity over the cached RoBERTa embeddings — return top-3 most similar training articles.
- **Stories served:** Demo audience
- **Acceptance criteria:**
  - Top-3 articles returned with `(headline, category, similarity_score)`.
  - Latency < 200 ms per query on Colab CPU.
- **Out of scope:** approximate-NN libraries (FAISS) — overkill at this scale.

#### F10 — Gradio demo app
Single-page UI on a Colab public share link.
- **Stories served:** Demo audience, Grader
- **Acceptance criteria:**
  - Two text inputs (headline, description); one "Predict" button.
  - Output region shows: predicted category, confidence (%), LLM explanation, list of 3 similar articles.
  - Public share link printed in the notebook output.
  - End-to-end inference completes in < 5 s on a warm Colab GPU runtime (after first call).
- **Out of scope:** persistent hosting, theming beyond Gradio defaults, multi-page navigation.

#### F11 — Documentation & deliverables
README, structured Jupyter notebook, `requirements.txt`, slide deck, plain-language written report.
- **Stories served:** Grader (completeness)
- **Acceptance criteria:**
  - README includes: overview, setup steps, API-key instructions, how to run, file map.
  - Notebook is code-focused with section headers — no prose-heavy markdown blocks.
  - `requirements.txt` pinned to exact versions used in Colab.
  - Slides explain the idea with diagrams; **no code on slides**.
  - Written report explains the concept in plain language for a non-ML audience.
- **Out of scope:** screencasts, video walk-throughs.

---

### v1.1 / Near-term (only if spare capacity remains after MVP)

- **F12 — Confidence calibration** for SVM classifiers via `CalibratedClassifierCV` so probabilities shown in the UI are meaningful. *Acceptance:* confidence histogram is well-spread, not all near 1.0 or 0.0.
- **F13 — Top-3 prediction display** in Gradio (instead of single label). *Acceptance:* UI shows top 3 categories with their probabilities side by side.

### Post-MVP (explicitly deferred)

- Permanent hosting on Hugging Face Spaces.
- Multilingual support.
- Online learning / drift detection.
- Fine-tuning RoBERTa end-to-end.
- Comparison with prompt-only LLM classification.

---

## 6. Tech Stack

| Layer | Choice | One-line rationale |
|---|---|---|
| Language / runtime | Python 3.10+ on Google Colab (GPU) | Spec mandates Colab Edition |
| Data + cleaning | `pandas`, `nltk`, `WordNet`, `re` | Standard NLP toolkit; spec names NLTK and WordNet explicitly |
| Classical features | `scikit-learn` (TF-IDF, scalers, sparse hstack) | Battle-tested, sparse-friendly |
| Classical models | `scikit-learn` (LogReg, LinearSVC, KNN, DT, RF, AdaBoost) | All six required by spec |
| Tuning | `GridSearchCV` / `RandomizedSearchCV` | Spec names them |
| Imbalance | `class_weight='balanced'` | Robust at 42 classes; SMOTE rejected on memory grounds (see ADR-002) |
| Deep embeddings | `transformers` + `roberta-base` + `torch` | Standard HF stack |
| LLM | Groq Cloud, model `llama-3.3-70b-versatile` | Free tier, fast, sufficient quality for explanations |
| Frontend | `gradio` 4.x | Mandated by spec; Colab share link supported natively |
| Persistence | `joblib` for models, `numpy`/`parquet` for data | Standard |
| Source control | Git + GitHub, feature-branch-per-task | 7-person workflow |

Decisions logged in `docs/decisions.md` (ADR-001 to ADR-006).

---

## 7. Architecture

See `docs/architecture.md`. Summary: a flat ML pipeline (no DB, no API server), with two parallel feature pipelines (TF-IDF + numeric, RoBERTa embeddings) feeding seven trained classifiers, evaluated by a shared metrics suite, and exposed via a Gradio app that adds an LLM explanation layer and similarity retrieval at inference time.

---

## 8. Non-Functional Requirements

### Performance
- End-to-end notebook re-run on Colab GPU runtime: **< 60 minutes** (cold; uses cached embeddings on warm runs).
- Single-prediction Gradio latency: **< 5 s** including LLM call.
- Cached prediction (same text): **< 1 s**.

### Security & secrets
- No credentials in code or git.
- `.gitignore` covers `kaggle.json`, `.env`, `data/`, `models/`, `*.npy`, `*.joblib`.
- Both API keys exposed during planning are treated as **compromised and rotated** before any use.

### Privacy & compliance
- The dataset is publicly licensed on Kaggle; no PII handling required.
- The Gradio public link is demo-only and torn down after grading.
- No user input is logged or stored.

### Accessibility
- Best effort within Gradio defaults; baseline keyboard navigation works. No formal WCAG audit (out of scope for academic delivery).

### Scalability
- Designed for the fixed dataset (~200K rows). No design constraints for streaming or growth.

### Reliability
- The notebook must complete without manual intervention given a valid `kaggle.json` and `GROQ_API_KEY`.
- Idempotent: re-running cells does not duplicate artefacts or break state.

### Observability
- `logging.INFO` in all `src/` modules; `tqdm` progress bars for long loops.
- Final cell of the notebook prints a one-screen summary: dataset size, split sizes, per-model metrics, time budget used.

---

## 9. Release Milestones

| Milestone | Target end | Definition of done |
|---|---|---|
| **M0 — Vertical slice running** | End of Sprint 1 | Data loaded; cleaning pipeline live; baseline LogReg trained on TF-IDF; minimal Gradio shell takes input and returns the LogReg prediction; repo skeleton + CI lint/test passing. |
| **M1 — Feature complete** | End of Sprint 2 | All 6 classical models trained and tuned; full evaluation suite produces comparison table + confusion matrices; RoBERTa embeddings cached; LinearSVC on embeddings trained; Groq LLM explanation working; similarity retrieval working; Gradio app shows all four output components. |
| **M2 — Polished delivery** | End of Sprint 3 | Notebook end-to-end clean; README finalised; slides finalised (no code); written report finalised; demo rehearsed; final tag cut on `main`. |

Mapping to sprints is in `docs/implementation-plan.md`.

---

## 10. Open Questions

- **Q1.** Will the team have access to Colab Pro (longer GPU sessions) or only the free tier? Affects fallback plan for embedding extraction. **Default assumption:** free tier, with subsampling fallback.
- **Q2.** Submission format expected by the instructor — single ZIP, GitHub link, or both? **Default assumption:** GitHub link + ZIP for safety.
- **Q3.** Is presentation in Arabic, English, or mixed? **Default assumption:** mixed (Arabic narration, English slides) — confirm with team lead before slide work begins.

---

## 11. Change Log

| Date | Change | Reason |
|---|---|---|
| 2026-05-08 | Initial PRD drafted | Project kickoff |
