# Demo Runbook — News Category Classification

This is the team's playbook for the 10-minute live demo + rehearsals **S3-T8** (rehearsal #1) and **S3-T10** (rehearsal #2).

It contains four things, in this order:

1. **The 10-minute script** — slide-by-slide timing with role assignments and what each speaker says.
2. **Pre-demo setup checklist** — what to verify in the 30 minutes before the live demo.
3. **Fallback plan** — what to do when something breaks live.
4. **Rehearsal notes template** — copy-paste form for the team to fill in during each rehearsal.

A separate **backup screencast** (not in this file) is recommended: a 5-minute video of the Gradio app running through all three pre-baked examples, recorded once and kept ready as the absolute last resort if the live link is unreachable. Recording tool: any (OBS / Windows Game Bar `Win+G` / Loom). Save as `docs/demo-screencast.mp4` and confirm it plays in PowerPoint's *Insert → Video → This Device* before the demo.

---

## 1. The 10-minute script

Total budget: **10 minutes**. Slides 1-15 are paced to land on the live-demo slide (§11) at the four-minute mark. The live demo itself takes 2.5 minutes; everything else is talk over slides.

| Time     | Slide                            | Speaker | What to say (paraphrase, do not memorise)                                                                                                                                                                                                                                                                                  |
| :------- | :------------------------------- | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0:00     | 1. Title                         | A       | "Hi, we're a 7-person team from Level 4 Data Science. We built a system that reads a news headline + description and predicts which of 27 categories it belongs to. Plus an LLM that explains the prediction. Live demo halfway through."                                                                                  |
| 0:30     | 2. The Problem                   | A       | "27 categories, multi-class. Inputs are short — a headline plus a paragraph. The output is one label, with a confidence score and a short rationale. Categories were originally 42 in the dataset; we consolidated them, more on that in two slides."                                                                       |
| 1:00     | 3. The Dataset                   | B       | "Kaggle's News Category Dataset v3 — 209 thousand articles from the *Huffington Post*. The big stat is the imbalance: largest class is 35 times bigger than the smallest. We addressed that with `class_weight='balanced'` on every classifier that supports it."                                                          |
| 1:30     | 4. Methodology                   | B       | "Six pipeline stages. Load from Kaggle, clean the text, consolidate categories from 42 to 27, build features two ways — TF-IDF for the classical models and RoBERTa embeddings for the deep model — train, then serve through Gradio."                                                                                     |
| 2:00     | 5. Preprocessing                 | C       | "Five-step cleaning. Lowercase, strip URLs and noise, tokenise with NLTK, drop stopwords, lemmatise with WordNet. Single function, used identically at training and at inference time. Plus three numeric features per article — word count, character count, punctuation count."                                          |
| 2:30     | 6. Category Consolidation        | C       | "The single biggest accuracy win in the project. Several of the original 42 labels were near-duplicates — three different *Arts* labels, two *Style* labels, three labels for political world news. Merging them lifted classical accuracy by about 10 points and RoBERTa by similar."                                     |
| 3:00     | 7. Six Classical Models          | D       | "Each trained with cross-validated hyper-parameter tuning on the 27-class data. Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, AdaBoost. Each one persisted to a `.joblib` file under `models/`."                                                                                                       |
| 3:30     | 8. RoBERTa + Confusion Matrix    | D       | "Our best model: a fine-tuned RoBERTa with a 27-way head. 75 % accuracy, macro-F1 of 0.71. The confusion matrix on the right shows where it still makes mistakes — three pairs dominate: Politics ↔ U.S. News, Entertainment ↔ Comedy, Health ↔ Family. These reflect genuine ambiguity in how the dataset was tagged."   |
| 4:00     | 9. Model Comparison              | E       | "All seven models ranked by macro-F1. RoBERTa wins clearly. Linear SVM is the best classical model at 69 %. AdaBoost essentially failed — it collapsed to predicting the majority class only. All four PRD acceptance bars at the bottom are met."                                                                          |
| 4:30     | 10. EDA Highlights               | E       | "Quick tour of the data. Severe imbalance, average article length is 29 words, vocabulary capped at 50 K tokens. Top words and bigrams reflect the dataset's news-of-the-day flavour — Trump, Covid, White House."                                                                                                          |
| 5:00     | 11. Live Demo                    | F       | "Demo time. Switching to the Gradio link." **(see §1.1 below for the live-demo script)**                                                                                                                                                                                                                                   |
| 7:30     | 12. LLM Explanation Layer        | G       | "We added one more layer: a Llama 3.3 70B model on Groq generates a two-or-three-sentence rationale for each prediction. Cached so repeated demo queries hit cache. If Groq fails, the UI degrades gracefully — explanation shows a placeholder, the rest of the prediction still renders."                                |
| 8:00     | 13. Lessons Learned              | G       | "Three honest lessons. First, cleaning is model-specific — feeding heavily-cleaned text to RoBERTa dropped accuracy from 0.75 to 0.58 until we caught it. Second, free-tier Colab GPU disconnects mid-run; we worked around it with shorter sessions and account-switching. Third, hybrid notebook-first scaled to 7 contributors when section ownership was clear." |
| 8:45     | 14. Future Work                  | A       | "Out of scope for this delivery: top-3 prediction display, confidence calibration, training RoBERTa from scratch for comparison, permanent Hugging Face hosting, multilingual support."                                                                                                                                       |
| 9:15     | 15. Closing / Q&A                | A       | "Thank you. Repository link is on the slide. Questions are welcome."                                                                                                                                                                                                                                                       |
| 9:30     | —                                | All     | Q&A — leave 30-90 seconds; do not run over.                                                                                                                                                                                                                                                                                  |

**Role legend** — A through G are seven speakers. Pair speakers up if the team prefers two co-presenters per section. The default split assumes one person per slide cluster but the team can re-shuffle as long as every section has an owner.

### 1.1 Live demo script (slides 5:00 - 7:30)

**Pre-staged Gradio link is open in a separate browser tab before the slide is reached.**

1. **Switch to the Gradio tab.** Show the empty form for ~3 seconds so the audience sees the UI shape — two text boxes, one button, four output regions.

2. **Example 1 — Tech (~30 sec):**
   - Headline: `Apple unveils new iPhone with improved camera`
   - Description: `The new model focuses on low-light photography and an upgraded sensor.`
   - Click *Predict*. Expected: **SCI_TECH**, ~98 % confidence, 2-3 sentence Groq explanation, top-3 similar articles all about Apple / iPhones.
   - Read the explanation aloud to the audience. Highlight that the similar articles are all genuinely related.

3. **Example 2 — Sports (~30 sec):**
   - Headline: `Lakers beat Celtics in overtime thriller`
   - Description: `LeBron James scored a career-high 47 points to lead the Lakers to victory.`
   - Click *Predict*. Expected: **SPORTS**, high confidence, Groq notes the basketball context.

4. **Example 3 — A harder one (~45 sec):**
   - Headline: `Senate passes sweeping climate bill`
   - Description: `The bill provides $369 billion in clean energy investments and tax credits.`
   - Click *Predict*. Expected: either **ENVIRONMENT** or **POLITICS_WORLD** — the model genuinely splits on this kind of story. Whatever it picks, the explanation will be coherent. Use this to show that the system *and* the LLM-explanation cope with ambiguity.

5. **Wrap (~30 sec):** Switch back to slide 11 in PowerPoint. Verbally: "That was the full pipeline end-to-end — prediction in under five seconds including the LLM call. Now slide 12 covers how the explanation gets generated."

**Hard time-out: leave slide 11 by 7:30** even if a prediction is still loading. The runbook has fallbacks below for the *prediction-too-slow* case.

---

## 2. Pre-demo setup checklist

Run through this in the 30 minutes before the live demo. Two team members work the list together — one types, one reads off the items.

### T-30 minutes — environment

- [ ] Open the live Sprint-2 Colab notebook (URL in `README.md`) on a laptop that will drive the demo.
- [ ] Confirm Runtime → GPU is connected (T4 visible in the top-right corner).
- [ ] In Colab Secrets, confirm three keys are set and "Notebook access" is on for each: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GROQ_API_KEY`.
- [ ] Run the §1 bootstrap cell. Wait for the success log line.
- [ ] Run §2 through §14 *Run all below* (Runtime → Run cell and below from the §1 cell). Expect ~10 minutes warm-cache, ~30 minutes cold.
- [ ] Confirm §14 prints a `https://*.gradio.live` public link.
- [ ] **Open that public link in a fresh browser tab.** Do not close this tab for the rest of the day.

### T-15 minutes — warm the demo path

Run all three live-demo examples through the public Gradio tab. This warms the LLM cache, the retrieval-embedding cache, and the model-prediction path — the second time the demo audience sees these examples, response will be < 1 second.

- [ ] Example 1 (Apple iPhone) — predicted, explanation generated, similar articles populated.
- [ ] Example 2 (Lakers Celtics) — predicted, explanation generated.
- [ ] Example 3 (climate bill) — predicted, explanation generated. Note which category it landed on (POLITICS_WORLD or ENVIRONMENT) — the speaker will mention this.

### T-5 minutes — display

- [ ] Slides open in PowerPoint, slideshow mode primed at slide 1.
- [ ] Browser tab with public Gradio link open in a separate display or as Alt-Tab target.
- [ ] Phone / second laptop has a backup of: the slides PDF export, the public Gradio link, and the screencast file (if recorded).
- [ ] Notifications silenced (Slack / Teams / Mail). Phone on Do Not Disturb.

---

## 3. Fallback plan — what to do when something breaks live

Five known failure modes and the immediate response. The principle: **never freeze on a broken thing**. Move on visibly and explain the substitute, do not silently retry.

| Failure                                              | Response                                                                                                                                                                                                                                                                                                                       |
| :--------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Public Gradio link 404 / dead.**                   | Switch to the screencast (`docs/demo-screencast.mp4`) embedded in slide 11 of the slides backup deck. Say: "The free-tier link expired between rehearsal and now — here's the recording from this morning's run." Continue narrating over the video. Total time hit: 0 seconds extra.                                          |
| **Groq rate-limited / down.**                        | The UI already handles this — explanation shows the placeholder string. Say to the audience: "Groq's rate-limited us — the explanation is showing the fallback message. The classifier and the similar-article retrieval still work, which is the rest of what the demo shows."                                                  |
| **Prediction takes > 8 seconds.**                    | Switch tabs visibly to PowerPoint slide 12 *while it's loading*. Say: "Cold-runtime first prediction takes a few seconds — let's cover the LLM layer while it warms up." Switch back when slide 12 ends. Never wait silently.                                                                                                  |
| **Colab kernel disconnected mid-demo.**              | Stop the live demo. Switch to the screencast. Reconnection on Colab free-tier in a live-demo window is not realistic — do not attempt. Total time hit: ~10 seconds for the tab switch.                                                                                                                                       |
| **Wi-fi drops on the demo machine.**                 | The presentation continues from slides only — slides 11 and 12 already describe the demo content. Say: "We've lost network — the static demo screenshots on these slides are from this morning's run." Skip the live portion. Total time hit: 0 seconds extra (the slides already exist).                                       |

The team's only standing rule: **the speaker who is currently presenting the slide is the one who calls the fallback**. They do not wait for someone behind the laptop to surface an error message.

---

## 4. Rehearsal notes template

Copy this section into a new file (`docs/rehearsal-1-notes.md` for S3-T8, `docs/rehearsal-2-notes.md` for S3-T10) and fill it in during the rehearsal. After rehearsal #1, the items in *Issues* feed directly into S3-T9 (bug-fix sweep).

```markdown
# Rehearsal #N — YYYY-MM-DD HH:MM

**Setting:** room / online / mixed
**Attendees:** A, B, C, D, E, F, G
**Driver of the demo machine:** X

## Timing

| Slide | Planned | Actual | Delta | Notes |
| :---- | :------ | :----- | :---- | :---- |
| 1     | 0:00    |        |       |       |
| 2     | 0:30    |        |       |       |
| 3     | 1:00    |        |       |       |
| 4     | 1:30    |        |       |       |
| 5     | 2:00    |        |       |       |
| 6     | 2:30    |        |       |       |
| 7     | 3:00    |        |       |       |
| 8     | 3:30    |        |       |       |
| 9     | 4:00    |        |       |       |
| 10    | 4:30    |        |       |       |
| 11 (demo start) | 5:00 |   |       |       |
| 11 (demo end)   | 7:30 |   |       |       |
| 12    | 7:30    |        |       |       |
| 13    | 8:00    |        |       |       |
| 14    | 8:45    |        |       |       |
| 15    | 9:15    |        |       |       |
| End   | 9:30    |        |       |       |

**Total run:** _______ (target 9-11 minutes)

## Issues found

For each issue, write: severity (S0 blocking / S1 should-fix / S2 nice-to-have), location (slide N / demo / runbook / README / notebook), one-line description, and proposed fix. These rows feed S3-T9.

| # | Severity | Where | Issue | Proposed fix |
| :- | :- | :- | :- | :- |
| 1 |   |   |   |   |
| 2 |   |   |   |   |

## What worked well

(Plain bullets — keep these too; they tell us what *not* to change in S3-T9.)

-

## Decisions taken in the room

-
```

---

## 5. Two checks before declaring rehearsal #1 complete (S3-T8 acceptance)

- [ ] The full run completed in **9 to 11 minutes** end-to-end.
- [ ] The *Issues* table in `docs/rehearsal-1-notes.md` is filled in (or contains a single row stating "no issues found").

When both boxes are ticked, post the rehearsal-1 notes to the team channel and ping me — that is the trigger for S3-T9 (bug-fix sweep).
