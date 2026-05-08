# Project Progress

## Status
- **Current milestone:** M0 — Vertical Slice
- **Current sprint:** Sprint 1 — Foundation & Vertical Slice (in progress)
- **Last updated:** 2026-05-08

## Completed Sprints
_none yet_

## Completed Tasks
- [x] **S1-T1** [2026-05-08] GitHub repo + `.gitignore` + branch protection. Repo at https://github.com/Omar-Anwar-Dev/news-category-classification (public, default branch `main`). Verification: `git status --ignored` confirms `.env` / `kaggle.json` / `.claude/` excluded; branch-protection API returns `required_approving_review_count: 1`, `allow_force_pushes: false`, `required_conversation_resolution: true`.
- [x] **S1-T2** [2026-05-08] `requirements.txt` pinned with 21 dependencies (Python 3.10+). Includes the 14 stack packages from the plan plus `numpy`, `scipy`, `pyarrow`, `tqdm`, `python-dotenv`. Syntactic validation: file parses as valid pip-requirements format. Full install verification is deferred to S1-T6 setup (when the venv goes up to run pytest).
- [x] **S1-T3** [2026-05-08] Scaffold per ADR-009 hybrid layout. Created `src/__init__.py`, `src/preprocessing.py` (stub), `tests/__init__.py`, `tests/test_smoke.py`, `notebooks/00_main.ipynb` (skeleton with 9 section headers covering S1-T5..T13), `notebooks/01_eda.ipynb` (skeleton), `pyproject.toml` (ruff + pytest config), 7 `.gitkeep` markers under `data/{,raw,processed,embeddings}/`, `models/`, `reports/{,confusion}/`. Updated `.gitignore` with negation patterns so `.gitkeep` files slip through `data/**/*` and `models/**/*` ignores. Verification: `python -c "import src"` returns OK; both notebooks parse as valid JSON; all 7 `.gitkeep` files report `TRACKED` under `git check-ignore`.
- [x] **S1-T4** [2026-05-08] GitHub Actions workflow at `.github/workflows/ci.yml`. Triggers on `pull_request` and `push` to `main`. Steps: checkout → setup Python 3.10 → install ruff/pytest/nltk → cache + download NLTK corpora (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) → `ruff check .` → `ruff format --check .` → `pytest -v`. Verified locally; first cloud run (`Lint + smoke tests` job, run 25531776401) green in 15s with all 12 steps ✓. Branch protection updated to make `Lint + smoke tests` a required status check before merge.
- [x] **S1-T5** [2026-05-08] `load_dataset()` cell implemented in `notebooks/00_main.ipynb`. Reads `kaggle.json` via `KAGGLE_CONFIG_DIR=PROJECT_ROOT`, downloads `rmisra/news-category-dataset` to `data/raw/News_Category_Dataset_v3.json` (~83 MB), parses JSON-Lines into a DataFrame. Verification (executed locally with the venv): first call downloaded in 12.5s, second call hit cache in 1.06s. **Row count: 209,527** (≥ 200K target). All six required columns present: `link, headline, category, short_description, authors, date`.
- [x] **S1-T6** [2026-05-08] (medium-risk) `src/preprocessing.py` implements full pipeline (lowercase → URL/HTML/punct/digit removal → NLTK tokenize → stopword removal → WordNet noun-then-verb lemmatisation), plus `build_numeric_features()` for word/char/punct counts on raw text. Idempotent NLTK corpora bootstrap on import. `tests/test_preprocessing.py` has the 6 mandated tests (order, empty, URL, lemma, stopword, numeric). Local verification: ruff lint + format clean; **8/8 pytest green** in 3.42s. End-to-end run on the full dataset: 209,527 articles cleaned in 34.5s (6,400 rows/sec); `data/processed/cleaned.parquet` (16.3 MB) cached for downstream tasks. Sample row: `[U.S. NEWS] "million american roll sleeve omicron target covid booster..."`.
- [x] **S1-T7** [2026-05-08] EDA part 1 in `notebooks/01_eda.ipynb`. Five cells: setup (loads both raw + cleaned frames), shape/dtypes/missing/duplicates/numeric-stats, findings markdown, class distribution + horizontal bar chart, class-imbalance commentary. Findings recorded from a real local run: 209,527 articles × 6 raw cols; missing concentrated in `authors` (17.9%) and `short_description` (9.4%); 13 row dupes + 1,531 duplicate headlines (kept for sprint 1); cleaned text averages 29 words / 174 chars. **42 classes, max-to-min ratio 35.1×** (POLITICS = 35,602 vs EDUCATION = 1,014); top-3 cover ~35% of corpus. Commentary references ADR-002 (`class_weight=balanced`) and the PRD §2 macro-F1 = 0.55 target.
- [x] **S1-T8** [2026-05-08] Classical features cell in `notebooks/00_main.ipynb`. Stratified 80/20 split (167,621 train / 41,906 test) with `random_state=42`. `TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)` fit on **train only** per PRD F4. `StandardScaler` over (word_count, char_count, punct_count) also train-only. Sparse hstack of TF-IDF + scaled numerics. Persisted: `models/label_encoder.joblib`, `models/tfidf_vectorizer.joblib`, `models/numeric_scaler.joblib`. **Verification:** `X_train.shape == (167621, 50003)`, `X_test.shape == (41906, 50003)` (matches the (N, 50_003) target); ~22 non-zeros per row; full pipeline ran in 14s. **Round-trip test passed** (load(saved).transform == original.transform on 5 sample rows).
- [x] **S1-T9** [2026-05-08] (medium-risk) LogReg baseline cell in `notebooks/00_main.ipynb`. As written: `LogisticRegression(class_weight='balanced', solver='saga', max_iter=300, random_state=42)` wrapped in `GridSearchCV(cv=3, scoring='f1_macro', n_jobs=-1)` over `C ∈ {0.1, 1.0, 10.0}` — runs as-is on Colab (~15-30 min on T4). Persists best estimator to `models/logreg_best.joblib` (16 MB) and asserts macro-F1 ≥ 0.45. **Local verification:** to keep within the user's chosen execution mode (b), I ran a single C=1 fit on the full 167,621-row train set (5.6 min, saga solver). Result on the held-out test set: **macro-F1 = 0.4744**, weighted-F1 = 0.5652 — comfortably above the 0.45 acceptance bar. `models/logreg_best.joblib` saved from this fit; the team's Colab run with full GridSearchCV will overwrite it with the tuned best, very likely improving these numbers further.

## In Progress
- [ ] **S1-T10** `evaluate_model()` cell — full metric suite + comparison CSV + confusion-matrix PNG

## Next in Sprint
- S1-T3 — Scaffold `src/` package + folder layout (hybrid: only `src/preprocessing.py`, rest in notebook)
- S1-T4 — GitHub Actions CI (ruff + pytest)
- S1-T5 — `data_loader.load_dataset()` with Kaggle download + cache
- S1-T6 — `preprocessing.clean_text()` + 6 unit tests
- S1-T7 — EDA part 1 notebook
- S1-T8 — `features.build_classical_features()` (TF-IDF + numeric)
- S1-T9 — `train_ml.train_baseline()` — LogReg with GridSearchCV
- S1-T10 — `evaluate.evaluate_model()` skeleton + comparison CSV
- S1-T11 — RoBERTa feasibility spike
- S1-T12 — Minimal Gradio app wired to LogReg baseline
- S1-T13 — Groq client smoke test
- S1-T14 — M0 sign-off — checklist + tag

## Blockers / Open Questions
_none_

## Risk Log Updates
_none yet_

## Notes
- Sprint 1 capacity: ~63h estimated of ~80h budget — comfortable.
- Execution model agreed: code + lighter parts run locally; GPU-bound tasks (RoBERTa spike, full embedding extraction) deferred to Colab.
- Architecture override: hybrid notebook-first per ADR-009 (supersedes ADR-005). Only `src/preprocessing.py` + `tests/` are kept as separate Python files. All other code lives in `notebooks/00_main.ipynb` cells.
