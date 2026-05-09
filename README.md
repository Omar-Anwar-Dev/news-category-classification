# News Category Classification

End-to-end multi-class text classifier that assigns one of **27 consolidated news categories** to a news article from its **headline** and **short description**.

Built as the deliverable for an academic Data Science course (Level 4) by a 7-person student team. Ships with six tuned classical models, a fine-tuned RoBERTa classifier, and a Gradio demo UI that adds Groq-hosted LLM explanations and similar-article retrieval.

| | |
| --- | --- |
| **Dataset** | [News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) on Kaggle (~209,000 rows). Categories consolidated from 42 → 27 (rationale in `docs/decisions.md`, ADR-010). |
| **Best model** | Fine-tuned `RobertaForSequenceClassification` over the 27 classes — **accuracy 0.7477, macro-F1 0.7102, weighted-F1 0.7562, ROC-AUC 0.9854**. |
| **Runtime** | Google Colab (free-tier T4 GPU); also runs locally on a Python 3.10+ venv (CPU-only, slower). |
| **License** | Academic — internal use only. |

---

## Quick start

The fastest path from clone to a working Gradio link is on Colab. Allow ~10 minutes on first run, ~3 minutes on cached re-runs.

### Option A — Colab (recommended)

The fastest entry point is the live notebook from the M1 run:

> **Live Colab notebook:** [colab.research.google.com/drive/18BokLALEH6-KKExzlJKy2TMdkOy2MWSV](https://colab.research.google.com/drive/18BokLALEH6-KKExzlJKy2TMdkOy2MWSV?usp=sharing)
>
> **Drive folder with all M1 artefacts** (model weights, embeddings, comparison CSV, confusion-matrix PNGs): [drive.google.com/drive/folders/1xFP6oExKXu6o6zC_uZSY1PZ4NMjx3FKR](https://drive.google.com/drive/folders/1xFP6oExKXu6o6zC_uZSY1PZ4NMjx3FKR?usp=sharing)

To run it yourself end-to-end:

1. Open the live Colab link above, **or** open `notebooks/00_main.ipynb` from this repo via the *Open In Colab* badge at the top of the file.
2. Connect to a **GPU runtime**: `Runtime → Change runtime type → T4 GPU → Save`.
3. Add three Colab Secrets (sidebar → key icon → "+ Add new secret", toggle "Notebook access" on for each):
   - `KAGGLE_USERNAME` — your Kaggle username
   - `KAGGLE_KEY` — your Kaggle API key (open `kaggle.json` from `https://www.kaggle.com/settings/account`)
   - `GROQ_API_KEY` — from [console.groq.com/keys](https://console.groq.com/keys)
4. **Runtime → Run all**. The bootstrap cell clones the repo, installs `requirements.txt`, downloads NLTK corpora, and pulls credentials from Colab Secrets.
5. The notebook ends with a public Gradio share link printed by the last code cell. Click it to open the demo. The link is valid for ~72 hours.

### Option B — local Python venv

```bash
git clone https://github.com/Omar-Anwar-Dev/news-category-classification.git
cd news-category-classification
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place `kaggle.json` (from your Kaggle account settings) at the project root, and create a `.env` file:

```bash
cp .env.example .env
# then edit .env to add your real GROQ_API_KEY
```

Open `notebooks/00_main.ipynb` in Jupyter / VS Code and **Run All**. CPU-only inference works for the classical models and the Gradio app; the fine-tuned RoBERTa runs on CPU but is ~10× slower than on a Colab T4 GPU.

---

## API keys you'll need

| Key | Where to get it | Where it's used |
| --- | --- | --- |
| Kaggle API token | [kaggle.com/settings/account → "Create New API Token"](https://www.kaggle.com/settings/account) downloads `kaggle.json` | Downloads the dataset |
| Groq API key | [console.groq.com/keys](https://console.groq.com/keys) (free tier sufficient for the demo) | Generates the LLM explanation behind each prediction |

Both keys are read at runtime — neither is committed to git. The `.gitignore` excludes `kaggle.json`, `.env`, the `data/` directory, and all model artefacts. If you accidentally commit either, **rotate the key immediately**.

---

## What the notebook does

`notebooks/00_main.ipynb` runs as 15 numbered sections, top-to-bottom:

| § | Stage | Output |
| --- | --- | --- |
| 1 | Data Loading | `data/raw/News_Category_Dataset_v3.json` |
| 2 | Preprocessing (lowercase → strip URLs/HTML/punct/digits → tokenise → stopword removal → WordNet lemmatise) + numeric features | `data/processed/cleaned.parquet` |
| 3 | Category consolidation 42 → 27 (ADR-010) | overwrites `cleaned.parquet`; updates `models/label_encoder.joblib` |
| 4 | Download fine-tuned RoBERTa from Google Drive (ADR-012) | `models/best_model/` |
| 5 | Pointer to `notebooks/01_eda.ipynb` (EDA Parts 1 + 2) | — |
| 6 | TF-IDF (uni+bi-gram, 50K features, sublinear) + numeric stack | `models/tfidf_vectorizer.joblib`, `models/numeric_scaler.joblib` |
| 7 | Logistic Regression baseline + GridSearchCV | `models/logreg_best.joblib` |
| 8 | Reusable `evaluate_model()` (accuracy, P/R/F1 macro+weighted, ROC-AUC OvR, confusion-matrix PNG) | `reports/model_comparison.csv`, `reports/confusion/<name>.png` |
| 9 | Five additional classical models with CV-tuning (LinearSVC, KNN, Decision Tree, Random Forest, AdaBoost) | five `models/<name>_best.joblib` files + comparison rows |
| 10 | Evaluate the fine-tuned RoBERTa on the held-out test set | row added to comparison table |
| 11 | Final 7-row comparison table, sorted by macro-F1, best-row highlighted | `reports/model_comparison.csv` |
| 12 | Groq LLM explanation function with `lru_cache` | — |
| 13 | Cosine-similarity retriever using the fine-tuned RoBERTa encoder | `data/embeddings/roberta_finetuned.npy`, `data/embeddings/index.parquet` |
| 14 | Gradio demo UI — predicted category + confidence + LLM explanation + top-3 similar articles | public share link printed in cell output |
| 15 | Final summary — dataset / split sizes / all model metrics / total wall-time | console output |

---

## Repository layout

```
news-category-classification/
├── notebooks/
│   ├── 00_main.ipynb       # Full pipeline + Gradio launch (15 sections)
│   └── 01_eda.ipynb        # EDA Part 1 + Part 2
├── src/
│   └── preprocessing.py    # clean_text() + numeric features (unit-tested)
├── tests/
│   ├── test_preprocessing.py
│   └── test_smoke.py
├── data/                   # gitignored — populated at runtime
├── models/                 # gitignored — populated at runtime
├── reports/
│   ├── model_comparison.csv
│   └── confusion/          # one PNG per model
├── docs/
│   ├── PRD.md              # product requirements
│   ├── architecture.md     # components, data flow
│   ├── implementation-plan.md
│   ├── decisions.md        # ADRs
│   ├── progress.md         # sprint status
│   ├── slides.pptx         # presentation deck
│   └── report.md           # written report (also report.docx)
├── .env.example
├── .gitignore
├── pyproject.toml          # ruff + pytest config
├── requirements.txt        # pinned dependencies (Python 3.10+)
└── README.md
```

Reusable Python lives **only** in `src/preprocessing.py` (because its `clean_text()` is unit-tested in CI). Every other pipeline step lives as cells in `notebooks/00_main.ipynb` — this is a deliberate hybrid notebook-first layout (ADR-009).

---

## Verifying the install (optional)

```bash
ruff check .
ruff format --check .
pytest -v
```

Expected: 8 / 8 tests green, lint + format clean. CI (`.github/workflows/ci.yml`) runs the same checks on every push.

---

## Team

Level 4 Data Science course project, 7-person student team, ~4-week timeline. The team coordinator's prior individual work on the same dataset contributed the fine-tuned RoBERTa weights used in §10 — see `docs/decisions.md` (ADR-011) for the full attribution. Course policy on reuse with attribution was confirmed at sprint-2 kickoff.

The team's contributions in this delivery cover: dataset preparation with category consolidation, six classical ML models trained and tuned from scratch, the full evaluation suite (metrics + confusion matrices + comparison table), comprehensive EDA across both notebook parts, Groq LLM integration with caching, similarity retrieval, the full Gradio interface, the written report, the slide deck, and the demo rehearsal.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `kaggle.json not found` on §1 | Credentials not provided | Add `KAGGLE_USERNAME` + `KAGGLE_KEY` as Colab Secrets, or place `kaggle.json` at the project root locally |
| §4 Drive download fails (`gdown` quota) | Drive throttling on the public link | Wait ~1 hour; or download `best_model.zip` manually from the Drive URL in `docs/decisions.md` (ADR-012) and unzip into `models/best_model/` |
| §10 RoBERTa accuracy ~0.58 instead of ~0.75 | RoBERTa is being fed the heavy-cleaned `text` instead of the lightly-cleaned `raw_text` | Verify the preprocessing cell (§2) wrote both columns; re-run §2 onwards |
| §14 Gradio share link doesn't open | Free Colab share link expired (~72 h) | Re-run the §14 cell to get a fresh link |
| `GROQ_API_KEY` errors during demo | Free-tier rate limit or missing key | The UI degrades gracefully — explanation shows a placeholder, prediction + similar articles still render |

---

## License

Academic project — internal use only. Not licensed for redistribution or commercial use.
