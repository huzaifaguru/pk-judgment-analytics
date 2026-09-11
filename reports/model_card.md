# Model Card — Outcome Classifier (Baseline)

**Task:** predict a Supreme Court judgment's outcome — `set aside`,
`allowed`, or `dismissed` — from structured case metadata plus a short
text snippet, evaluated with 5-fold stratified cross-validation.

## Read this before the numbers below

- Trained on **571 cases** — the ones with a real extracted
  outcome label (see `data_quality_report.md` §3). That is a small dataset
  for a 3-class problem, and two of the three classes have only
  **31** and **31**
  examples respectively.
- Classes: set aside = 509, dismissed = 31, allowed = 31. This
  is a **89% majority class** problem —
  a model that always guesses `set aside` gets that accuracy for free, so
  accuracy alone is not a meaningful score here; macro-F1 (below) weighs
  all three classes equally instead.
- The only text signal available is `petition_text`, which is the
  **first 20 lines of the extracted judgment**, not the actual petition —
  a limitation of the original extraction script, not this pipeline. Treat
  any contribution from that feature as weak, incidental signal (e.g.
  boilerplate opening phrasing), not genuine legal-argument understanding.
- With classes this small, cross-validation folds still carry meaningful
  sampling noise. These numbers describe what this specific pipeline does
  on this specific 571-case sample — **they are not a claim
  about real-world predictive reliability**, and should not be presented
  as one.

## Baseline comparison (same 5 folds, same data)

Majority-class dummy classifier (always predicts `set aside`):

```
              precision    recall  f1-score   support

     allowed       0.00      0.00      0.00        31
   dismissed       0.00      0.00      0.00        31
   set aside       0.89      1.00      0.94       509

    accuracy                           0.89       571
   macro avg       0.30      0.33      0.31       571
weighted avg       0.79      0.89      0.84       571

```
Macro-F1: 0.314

## This model (logistic regression, class-balanced, TF-IDF + structured features)

```
              precision    recall  f1-score   support

     allowed       0.15      0.32      0.20        31
   dismissed       0.09      0.23      0.13        31
   set aside       0.95      0.79      0.86       509

    accuracy                           0.74       571
   macro avg       0.40      0.45      0.40       571
weighted avg       0.86      0.74      0.79       571

```
Macro-F1: 0.399

## How to read the comparison

The model's macro-F1 (0.399) versus the dummy baseline's
(0.314) is the honest headline number — it shows whether the
model does better than always guessing the majority class at recognizing
the minority outcomes, not just whether its accuracy looks high (accuracy
is easy to inflate here by ignoring the minority classes entirely, which is
exactly what the dummy baseline does).

See `reports/figures/08_confusion_matrix.png` for the full breakdown of
where the model's predictions land.

## Reproducing this

```
python src/train_model.py
```
Uses a fixed `random_state=42` for the fold split, so re-running should
reproduce these exact numbers on an unchanged `clean_cases.csv`.
