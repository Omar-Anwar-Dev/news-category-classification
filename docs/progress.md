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
- [x] **S1-T4** [2026-05-08] GitHub Actions workflow at `.github/workflows/ci.yml`. Triggers on `pull_request` and `push` to `main`. Steps: checkout → setup Python 3.10 → install ruff/pytest/nltk → cache + download NLTK corpora (`punkt`, `punkt_tab`, `stopwords`, `wordnet`) → `ruff check .` → `ruff format --check .` → `pytest -v`. Verified locally with the project's `.venv` (Python 3.10.11): all 4 files clean against ruff lint + format; `pytest` collects 2 tests, both pass in 0.05s. Cloud CI run will be observed after the next push.

## In Progress
- [ ] **S1-T5** `data_loader.load_dataset()` cell — Kaggle download with local cache + JSON-Lines parse

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
