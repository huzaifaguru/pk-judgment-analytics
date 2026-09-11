#!/usr/bin/env python3
"""
train_model.py — one honest baseline classifier for judgment outcome,
with limitations disclosed up front rather than a majority-class-driven
accuracy number presented alone.

Input : data/processed/clean_cases.csv
Output: reports/model_card.md
        reports/figures/08_confusion_matrix.png

Design decisions (all documented in the model card too):
- Only the 571 cases with a REAL extracted outcome are used (see
  data_quality_report.md, section 3) — no case is trained on a made-up
  "unknown" label.
- 'appeal allowed' and 'leave to appeal granted' (n=1) are merged into a
  single 'allowed' class, and 'appeal dismissed' / 'petition dismissed'
  into 'dismissed', because stratified cross-validation cannot work with a
  1-member class. This gives 3 classes: set aside (509), allowed (31),
  dismissed (31).
- Evaluation is 5-fold stratified cross-validation with out-of-fold
  predictions (cross_val_predict), not a single train/test split — with
  minority classes this small, one split would be noise, not a result.
- A majority-class dummy baseline is reported side by side with the real
  model, on the exact same folds, so the model's numbers can be judged
  against the floor they need to beat rather than in isolation.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "clean_cases.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT = ROOT / "reports" / "model_card.md"

LABEL_MAP = {
    "set aside": "set aside",
    "appeal allowed": "allowed",
    "leave to appeal granted": "allowed",
    "appeal dismissed": "dismissed",
    "petition dismissed": "dismissed",
}

NUMERIC_FEATURES = ["word_count", "char_count", "num_sections", "year", "month"]
CATEGORICAL_FEATURES = ["case_type", "legal_category"]
TEXT_FEATURE = "petition_text"


def build_pipeline():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unclassified")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    text = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, stop_words="english")

    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("text", text, TEXT_FEATURE),
    ])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = pd.read_csv(DATA)
    labeled = df[df["outcome"].notna()].copy()
    labeled["target"] = labeled["outcome"].map(LABEL_MAP)
    labeled = labeled.dropna(subset=["target"])
    labeled[TEXT_FEATURE] = labeled[TEXT_FEATURE].fillna("")

    X = labeled[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
    y = labeled["target"]
    class_counts = y.value_counts()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = build_pipeline()
    y_pred = cross_val_predict(pipe, X, y, cv=cv)

    dummy = DummyClassifier(strategy="most_frequent")
    y_pred_dummy = cross_val_predict(dummy, X, y, cv=cv)

    labels = sorted(y.unique())
    report_txt = classification_report(y, y_pred, labels=labels, digits=2, zero_division=0)
    dummy_report_txt = classification_report(y, y_pred_dummy, labels=labels, digits=2, zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    dummy_macro_f1 = f1_score(y, y_pred_dummy, average="macro", zero_division=0)

    cm = confusion_matrix(y, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (5-fold out-of-fold predictions)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "08_confusion_matrix.png", dpi=150)
    plt.close()

    report = f"""# Model Card — Outcome Classifier (Baseline)

**Task:** predict a Supreme Court judgment's outcome — `set aside`,
`allowed`, or `dismissed` — from structured case metadata plus a short
text snippet, evaluated with 5-fold stratified cross-validation.

## Read this before the numbers below

- Trained on **{len(labeled)} cases** — the ones with a real extracted
  outcome label (see `data_quality_report.md` §3). That is a small dataset
  for a 3-class problem, and two of the three classes have only
  **{class_counts.get('allowed', 0)}** and **{class_counts.get('dismissed', 0)}**
  examples respectively.
- Classes: {', '.join(f"{k} = {v}" for k, v in class_counts.items())}. This
  is a **{class_counts.max()/len(labeled):.0%} majority class** problem —
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
  on this specific {len(labeled)}-case sample — **they are not a claim
  about real-world predictive reliability**, and should not be presented
  as one.

## Baseline comparison (same 5 folds, same data)

Majority-class dummy classifier (always predicts `{y.value_counts().idxmax()}`):

```
{dummy_report_txt}
```
Macro-F1: {dummy_macro_f1:.3f}

## This model (logistic regression, class-balanced, TF-IDF + structured features)

```
{report_txt}
```
Macro-F1: {macro_f1:.3f}

## How to read the comparison

The model's macro-F1 ({macro_f1:.3f}) versus the dummy baseline's
({dummy_macro_f1:.3f}) is the honest headline number — it shows whether the
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
"""
    REPORT.write_text(report)
    print(report)
    print(f"Wrote {REPORT}")
    print(f"Wrote {FIG_DIR / '08_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
