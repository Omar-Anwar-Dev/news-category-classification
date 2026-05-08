# Project Progress

## Status
- **Current milestone:** M0 — Vertical Slice
- **Current sprint:** Sprint 1 — Foundation & Vertical Slice (in progress)
- **Last updated:** 2026-05-08

## Completed Sprints
_none yet_

## Completed Tasks
- [x] **S1-T1** [2026-05-08] GitHub repo + `.gitignore` + branch protection. Repo at https://github.com/Omar-Anwar-Dev/news-category-classification (public, default branch `main`). Verification: `git status --ignored` confirms `.env` / `kaggle.json` / `.claude/` excluded; branch-protection API returns `required_approving_review_count: 1`, `allow_force_pushes: false`, `required_conversation_resolution: true`.

## In Progress
- [ ] **S1-T2** Pin baseline `requirements.txt` — starting next

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
