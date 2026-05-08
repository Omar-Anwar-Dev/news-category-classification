# Decision Log

Append-only record of non-trivial product and technical decisions made during planning. New entries go at the bottom.

---

## ADR-001: Use Groq Llama 3.3 70B for the LLM bonus
**Date:** 2026-05-08
**Status:** Accepted

**Context:** The spec requires the LLM bonus to use one cloud LLM among OpenAI / Groq / one other. The team needs a free, fast, and reliable provider; the LLM is used only for explanation generation (3 sentences per request), not classification.

**Decision:** Groq Cloud, model `llama-3.3-70b-versatile`.

**Alternatives considered:**
- OpenAI GPT-4o-mini — paid; demo cost concerns; rate limits unclear without billing.
- Other commercial APIs — paid; same concerns.

**Consequences:** Free tier is sufficient for development, demo, and grading. Latency on Groq is excellent (sub-second). The team must handle Groq-specific rate limits and the (small) chance of free-tier throttling on demo day; mitigated by caching all explanations and pre-warming the cache before the live demo.

---

## ADR-002: Handle class imbalance with `class_weight='balanced'`, not SMOTE
**Date:** 2026-05-08
**Status:** Accepted

**Context:** The dataset has 42 classes with severe imbalance — minority categories have <100 samples while majority categories exceed 30K. The spec offers SMOTE *or* `class_weight` as acceptable.

**Decision:** Use `class_weight='balanced'` for every classifier that supports it. Do not use SMOTE.

**Alternatives considered:**
- SMOTE on TF-IDF feature space — synthesises new sparse rows in a 50K-dimensional space across 42 classes; memory blow-up in Colab and questionable semantic validity for sparse text features.
- Random oversampling — same memory issue at smaller scale; doesn't help generalisation.
- Class-aware sampling at the batch level — overcomplicated for scikit-learn classical models.

**Consequences:** Solid, well-supported approach for sparse text classification. Macro-F1 is reported alongside accuracy and weighted-F1 to avoid masking minority-class performance. We accept that very-low-frequency classes will still be hard to learn — this is acknowledged in the written report.

---

## ADR-003: Use frozen RoBERTa embeddings + LinearSVC, not fine-tuned RoBERTa
**Date:** 2026-05-08
**Status:** Superseded by ADR-011 (2026-05-08)

**Context:** The spec's bonus pipeline diagram is "Sentence → BERT → 768-dim vector → SVM classifier" — pre-trained embeddings, not fine-tuning. The team has 1 month, a non-trivial classical workstream, and Colab free-tier GPU.

**Decision:** Use `roberta-base` in inference mode (no parameter updates), mean-pool the last hidden states, feed into `LinearSVC`. Cache embeddings to disk so they are computed once.

**Alternatives considered:**
- Fine-tuning RoBERTa for sequence classification — would likely outperform but costs days of GPU and is not what the spec asks for.
- DistilBERT — smaller and faster but a step down on quality; not specifically named in the spec.

**Consequences:** Predictable runtime, fits the spec exactly, cleanly separates the deep representation step from the classification step, and lets the team reuse the cached embeddings for similarity retrieval.

---

## ADR-004: Mean-pooling over RoBERTa hidden states (not the CLS token)
**Date:** 2026-05-08
**Status:** Accepted

**Context:** When using RoBERTa as a frozen feature extractor for sentence-level tasks, we have to pick a pooling strategy.

**Decision:** Mean-pool the last hidden layer's token embeddings (excluding padding) to produce a single 768-dim vector per article.

**Alternatives considered:**
- `[CLS]` token embedding — `roberta-base`'s `[CLS]` is not as semantically saturated as BERT's because RoBERTa removed the NSP objective; mean-pooling is the standard recommendation in the sentence-embeddings literature.
- Last-4-layer concatenation — quadruples dimensionality without clear gains for this task.

**Consequences:** Marginally better sentence representations than CLS for downstream classification, with no extra compute cost.

---

## ADR-005: Modular `src/*.py` + a thin notebook (not everything in the notebook)
**Date:** 2026-05-08
**Status:** Superseded by ADR-009 (2026-05-08)

**Context:** A 7-person team editing one large Jupyter notebook is a recipe for merge conflicts. The spec also explicitly asks for a `src/` package alongside the notebook.

**Decision:** Reusable code lives in plain `.py` modules under `src/`. The notebook imports from `src/` and only contains orchestration cells, EDA charts, and narrative.

**Alternatives considered:**
- Everything in the notebook — fast to start, painful to merge, hard to unit-test.
- Pure script with no notebook — would not satisfy the spec, which requires a structured Jupyter notebook.

**Consequences:** Easier parallel work and unit testing; small upfront cost to set up the package structure and the import path inside Colab.

---

## ADR-006: GitHub with feature-branch-per-task workflow
**Date:** 2026-05-08
**Status:** Accepted

**Context:** 7 students working in parallel for a month. We need a single source of truth for code and a way to coordinate.

**Decision:** GitHub repository, `main` is protected, every task gets its own short-lived `feature/<task-id>` branch, merged via pull request after at least one teammate's review.

**Alternatives considered:**
- Google Drive notebook sharing — no version history; concurrent edits silently overwrite.
- One shared Colab notebook — same problem.

**Consequences:** Adds a small Git learning overhead for any team member who hasn't used branches before. In return: full history, conflict resolution, code review, easy rollback. Worth it.

---

## ADR-007: Cap TF-IDF vocabulary at 50,000 features with `sublinear_tf=True`
**Date:** 2026-05-08
**Status:** Accepted

**Context:** Unbounded TF-IDF on 200K articles with uni+bi-grams produces ~1M features. That blows up memory and slows tuning, with little gain in classification accuracy past the top tens of thousands.

**Decision:** `TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True, min_df=2)`.

**Alternatives considered:**
- `max_features=100_000` — slightly higher recall on rare bi-grams; doubles memory.
- `HashingVectorizer` — no vocabulary tracking, harder to interpret, no inverse-document-frequency.

**Consequences:** Predictable memory footprint, fast model fitting, retains practically all discriminative features. `min_df=2` drops singleton terms (typos, hapaxes).

---

## ADR-008: `LinearSVC` instead of `SVC(kernel='linear')`
**Date:** 2026-05-08
**Status:** Accepted

**Context:** The spec asks for SVM. With ~160K training rows, kernel SVM (`SVC`) is impractically slow.

**Decision:** Use `sklearn.svm.LinearSVC` for the SVM model in the comparison table. Wrap it in `CalibratedClassifierCV` to expose `predict_proba` for the Gradio confidence display.

**Alternatives considered:**
- `SVC(kernel='linear')` — same decision boundary in theory, dramatically slower in practice on this scale.
- `SGDClassifier(loss='hinge')` — equivalent objective, harder to tune to the same accuracy.

**Consequences:** Training time drops from hours to minutes. We retain a true linear-SVM model in the comparison table, with calibrated probabilities for the UI.

---

## ADR-009: Hybrid notebook-first layout — supersedes ADR-005
**Date:** 2026-05-08
**Status:** Accepted; supersedes ADR-005

**Context:** ADR-005 originally chose a modular `src/*.py` package with a thin notebook to reduce 7-person merge conflicts and enable unit testing. During Sprint 1 kickoff the team coordinator requested all development happen inside the Jupyter notebook instead, finding the notebook-first style faster to iterate and easier for the team to read end-to-end.

**Decision:** Develop the entire pipeline (data loading, preprocessing scaffolding, EDA, classical feature engineering, model training, evaluation, RoBERTa embeddings, retrieval, LLM, Gradio launch) inside `notebooks/00_main.ipynb` as ordered cells. Carve out **only** `src/preprocessing.py` (and its sibling `tests/test_preprocessing.py`) as standalone Python files, because S1-T6 requires 6 unit tests on `clean_text()` running in CI, and CI cannot exercise functions that live solely in notebook cells.

**Alternatives considered:**
- Keep ADR-005 (full `src/` package) — rejected per coordinator's preference.
- Notebook-only with no `src/` at all — rejected because it kills S1-T4 CI and S1-T6 unit-test requirements without an alternative way to verify cleaning correctness; preprocessing bugs invalidate every downstream model (S1-T6 risk).
- Use `nbdev`/`jupytext` to auto-export notebook cells to `.py` — rejected as overkill for a 4-week academic project.

**Consequences:**
- Single-file notebook is easier to read end-to-end and matches grader expectations for "code-focused notebook" (PRD F11).
- Higher merge-conflict risk on the notebook with 7 contributors. Mitigation: section-ownership per teammate (each owns a contiguous block of cells), agreed cell order frozen early, and a pre-merge convention of running "Restart & Clear All Outputs" before commit so cell metadata stays minimal.
- Implementation plan S1-T3 is rescoped: instead of scaffolding 9 modules, create `src/preprocessing.py` + `src/__init__.py` + `tests/test_preprocessing.py` + `notebooks/00_main.ipynb` skeleton + the empty data/model/reports placeholder dirs.
- Downstream tasks (S1-T5, S1-T8 to S1-T13) implement their logic as cells in `00_main.ipynb` rather than as `src/*.py` modules. Acceptance criteria for those tasks remain functionally identical (same outputs, same artefacts on disk); only the file location changes.

---

## ADR-010: Category consolidation — 42 → 27 classes
**Date:** 2026-05-08
**Status:** Accepted

**Context:** The raw News Category Dataset has 42 labels, several of which are near-duplicates of each other (`ARTS` / `ARTS & CULTURE` / `CULTURE & ARTS`; `STYLE` / `STYLE & BEAUTY`; `WORLDPOST` / `THE WORLDPOST`; `PARENTS` / `PARENTING`; `WELLNESS` / `HEALTHY LIVING`; `TECH` / `SCIENCE`; etc.). Sprint 1's 42-class TF-IDF baseline reached 55% accuracy with a large fraction of errors being inter-near-duplicate confusions, visible directly in the confusion-matrix PNG. Prior individual work by the team coordinator on the same dataset showed that consolidating these duplicates lifts classical models from ~55% to ~66% accuracy and brings them past the spec's macro-F1 target.

**Decision:** Apply a fixed category remap before training. Final 27 classes:

```
ARTS, BLACK VOICES, BUSINESS, COMEDY, CRIME, DIVORCE, EDUCATION, ENTERTAINMENT,
ENVIRONMENT, FAMILY, FIFTY, FOOD, GOOD NEWS, HEALTH, HOME & LIVING, IMPACT,
LATINO VOICES, POLITICS_WORLD, QUEER VOICES, RELIGION, SCI_TECH, SPORTS, STYLE,
TRAVEL, U.S. NEWS, WEDDINGS, WORLD NEWS
```

Mapping:

```python
CATEGORY_MAP = {
    "STYLE & BEAUTY": "STYLE",
    "THE WORLDPOST": "POLITICS_WORLD", "WORLDPOST": "POLITICS_WORLD", "POLITICS": "POLITICS_WORLD",
    "WELLNESS": "HEALTH", "HEALTHY LIVING": "HEALTH",
    "TASTE": "FOOD", "FOOD & DRINK": "FOOD",
    "ARTS": "ARTS", "CULTURE & ARTS": "ARTS", "ARTS & CULTURE": "ARTS",
    "COLLEGE": "EDUCATION", "EDUCATION": "EDUCATION",
    "SCIENCE": "SCI_TECH", "TECH": "SCI_TECH",
    "GREEN": "ENVIRONMENT", "ENVIRONMENT": "ENVIRONMENT",
    "MONEY": "BUSINESS", "BUSINESS": "BUSINESS",
    "WOMEN": "FAMILY", "PARENTS": "FAMILY", "PARENTING": "FAMILY",
    "CRIME": "CRIME", "WEIRD NEWS": "CRIME",
    "MEDIA": "ENTERTAINMENT", "ENTERTAINMENT": "ENTERTAINMENT",
    "STYLE": "STYLE",
}
KEEP_AS_IS = {"SPORTS", "TRAVEL", "RELIGION", "COMEDY", "QUEER VOICES", "BLACK VOICES",
              "DIVORCE", "FIFTY", "GOOD NEWS", "HOME & LIVING", "IMPACT", "LATINO VOICES",
              "U.S. NEWS", "WEDDINGS", "WORLD NEWS"}
MIN_SAMPLES_PER_CLASS = 1000
```

Apply in addition: drop rows with cleaned text < 4 words; drop exact duplicates. Final dataset size: ~206,960 rows.

**Alternatives considered:**
- Keep all 42 — accuracy capped at ~55% baseline, ~62-65% with frozen RoBERTa. Below the team's 70% target.
- Auto-cluster categories by name embeddings — non-deterministic and harder to defend in the report.
- Hierarchical labelling (keep both fine + coarse) — out of scope; complicates training and inference.

**Consequences:** PRD §1 changes from "42 categories" to "27 consolidated categories"; defended in the written report by showing the inter-near-duplicate confusions in the sprint-1 confusion matrix. The 27-class formulation is the project's official problem statement from this point on. All models (classical + fine-tuned) train and report against the 27-class label space.

---

## ADR-011: Reuse fine-tuned RoBERTa from coordinator's prior work — supersedes ADR-003
**Date:** 2026-05-08
**Status:** Accepted; supersedes ADR-003

**Context:** ADR-003 originally chose frozen `roberta-base` embeddings + `LinearSVC` for the deep-model bonus, citing time and Colab GPU constraints. Discovered during sprint-1 closing review that the team coordinator has a fully fine-tuned `RobertaForSequenceClassification` model from a prior individual project on the same dataset, hitting ~70% top-1 accuracy on the 27-class merged setup. Re-training from scratch on Colab free tier (no Pro subscription) is infeasible within sprint 2 — a single 3-epoch fine-tune on 207K rows takes 8-15 hours of GPU and routinely hits Colab's idle / max-runtime disconnects. Frozen embeddings would only reach ~62-65% accuracy, below the 70% target.

**Decision:** Use the coordinator's existing fine-tuned model (`best_model/`, 498 MB, ~70% accuracy, classes match ADR-010's 27) as the project's primary classifier. The Gradio app's `BEST_MODEL` constant points to it; the model-comparison table includes it as the "RoBERTa-tuned" row alongside the six classical models the team trains from scratch.

**Alternatives considered:**
- Re-train RoBERTa from scratch with the team — risks Colab session timeouts; lower-quality model on a 50-100K subset; eats most of sprint 2's capacity.
- Stick with frozen RoBERTa + LinearSVC (original ADR-003) — gives ~62-65%; misses the 70% target.
- Skip the deep-model path entirely — sacrifices F7 bonus and a major part of the project's grade.
- Train on a subset (e.g. 80K rows, 2 epochs, distilroberta-base) — feasible on free tier in ~2-3 hours but yields ~67-69%; below target.

**Consequences:**
- F7 reframed: "Fine-tuned RoBERTa-base for sequence classification (27 classes)" instead of "frozen embeddings + LinearSVC".
- Methodology disclosed transparently in the written report and the slide deck: "the fine-tuned RoBERTa is the result of pre-project individual work by the team coordinator; the team's contributions in this delivery span dataset preparation with category consolidation, six classical ML models trained from scratch with hyper-parameter tuning, the full evaluation suite, comprehensive EDA part 2, Groq LLM integration with caching, similarity retrieval, the full Gradio interface, the written report, and demo rehearsal."
- Course policy on reuse with attribution confirmed by coordinator (syllabus reviewed sprint-2 kickoff, 2026-05-08).
- The full fine-tuning recipe (CATEGORY_MAP + Trainer hyper-parameters from `training_args.bin`) is documented in a markdown cell in the notebook for reproducibility, even though the team will not re-run it.
- PRD §2 success criterion 4 raised from "RoBERTa+SVC macro-F1 ≥ 0.60" to "Fine-tuned RoBERTa accuracy ≥ 0.70, macro-F1 ≥ 0.65" — based on the prior model's actual measured performance.

---

## ADR-012: Store fine-tuned model on Google Drive (download via gdown)
**Date:** 2026-05-08
**Status:** Accepted

**Context:** The fine-tuned model package (`best_model.zip`, ~470 MB) exceeds GitHub's 100 MB file limit and pushes against git-LFS's free-tier 1 GB cap. Need an accessible storage location any team member can pull from any Colab session, ideally with no extra account setup.

**Decision:** Store `best_model.zip` on the team coordinator's Google Drive at link visibility "Anyone with the link". The notebook's bootstrap downloads via `gdown` keyed by Drive `FILE_ID`, unzips into `models/best_model/`, then loads with `transformers.AutoModelForSequenceClassification.from_pretrained()`. Cached after first run (subsequent Colab sessions reuse the local extraction).

**FILE_ID:** `19EIWqmmR4tbJrMyiqKYRT__s_d1n11rW` (project-public; not a secret).

**Alternatives considered:**
- HuggingFace Hub — most idiomatic; free; supports versioning. Rejected only because the coordinator preferred Drive as already-familiar tool.
- Git LFS — complicates clone; hits free-tier 1 GB limit on first push; awkward for grader.
- Commit zipped model to repo — violates GitHub size limit.

**Consequences:**
- The Drive `FILE_ID` is committed to the notebook (project-public).
- Bootstrap adds ~1-2 minutes on the first cold Colab session to download + extract; cached after that.
- If Drive sharing is revoked or the file is deleted, the project breaks; mitigation: coordinator keeps the original `best_model/` folder locally as a backup and documents the upload steps so anyone can re-host.
- The model is downloadable by anyone with the notebook URL; acceptable for an academic deliverable but worth noting in the README.
