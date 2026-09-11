#!/usr/bin/env python3
"""
build_similarity_index.py — precomputes a TF-IDF similarity index over the
571 outcome-labeled cases, for the "similar past cases" search feature on
the dashboard.

This is a search feature, not a prediction feature: given a short
description of a situation, it returns the closest-matching past judgments
by text similarity, with their real outcome and sections cited attached.
It does not claim to predict anything.

Runs entirely client-side once built: this script exports the vocabulary,
IDF weights, and each case's TF-IDF vector as JSON, and docs/index.html
computes the query vector and cosine similarity in plain JS. No backend,
same static-hosting model as the rest of the dashboard.

Input : data/processed/clean_cases.csv
Output: docs/similarity_data.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "clean_cases.csv"
OUT = ROOT / "docs" / "similarity_data.json"

# Same scope as the classifier (see train_model.py / model_card.md): only
# cases with a real extracted outcome. petition_text is the first 20 lines
# of the judgment (an extraction-script constraint) — the same limitation
# noted throughout this repo applies here too.
MAX_FEATURES = 1500
MIN_DF = 2
SNIPPET_CHARS = 220


def main():
    df = pd.read_csv(DATA)
    labeled = df[df["outcome"].notna()].copy()
    labeled["petition_text"] = labeled["petition_text"].fillna("")
    labeled = labeled[labeled["petition_text"].str.strip() != ""].reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        stop_words="english",
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(labeled["petition_text"])
    vocab = vectorizer.get_feature_names_out().tolist()
    idf = vectorizer.idf_.tolist()

    tfidf_coo = tfidf.tocoo()
    doc_vectors = [[] for _ in range(tfidf.shape[0])]
    for row, col, val in zip(tfidf_coo.row, tfidf_coo.col, tfidf_coo.data):
        doc_vectors[row].append([int(col), round(float(val), 5)])

    def snippet(text):
        text = " ".join(text.split())
        return text[:SNIPPET_CHARS] + ("…" if len(text) > SNIPPET_CHARS else "")

    cases = []
    for i, row in labeled.iterrows():
        cases.append({
            "case_no": row.get("case_no") if pd.notna(row.get("case_no")) else None,
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "case_type": row.get("case_type") if pd.notna(row.get("case_type")) else None,
            "outcome": row.get("outcome"),
            "num_sections": int(row["num_sections"]) if pd.notna(row.get("num_sections")) else None,
            "word_count": int(row["word_count"]) if pd.notna(row.get("word_count")) else None,
            "snippet": snippet(row["petition_text"]),
            "vec": doc_vectors[i],
        })

    payload = {
        "vocab": vocab,
        "idf": [round(v, 5) for v in idf],
        "cases": cases,
        "note": "TF-IDF over petition_text (first 20 lines of each judgment, "
                "an extraction-script limit) for the 571 cases with a real "
                "extracted outcome. This is similarity search, not prediction.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT.stat().st_size / 1024
    print(f"Vocabulary: {len(vocab)} terms")
    print(f"Cases indexed: {len(cases)}")
    print(f"Wrote {OUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
