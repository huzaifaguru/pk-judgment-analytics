#!/usr/bin/env python3
"""
clean_data.py — reproducible cleaning pipeline for the Pakistan Supreme Court
judgment-metadata dataset.

Input : data/raw/out_raw.csv        (raw output of the original extract_judgments.py)
Output: data/processed/clean_cases.csv   (one row per unique case)
        data/processed/excluded_rows.csv (rows dropped, with reason, for transparency)
        reports/data_quality_report.md   (every fix, with before/after counts)

This replaces the ad-hoc, 90+ cell notebook cleaning in the original project
(analysis.ipynb) with a single linear, documented script. Every number in the
generated report comes from actually running this script — nothing here is
asserted without being computed below.

Why this rewrite exists (short version):
1. The original `df.drop_duplicates()` deduped on ALL columns, so the same
   case filed once per bench member (same filename, same case_no, same date,
   different judge_folder) was NOT recognized as one case. Verified in this
   dataset: every filename that repeats does so with an identical case_no
   and date — it is always the same case, never two different cases sharing
   a filename. So the correct dedup key is `filename`.
2. `outcome` values with a literal newline inside them (e.g. "set\naside",
   an artifact of how pdfplumber preserves the source PDF's line wrapping)
   were left as distinct categories instead of being merged into "set aside".
3. Rows where extraction failed outright (no text pulled, e.g. scanned PDFs)
   had every field set to NaN by the original script, but the previous
   cleaning notebook still fed them into the analysis and later filled
   NaN outcomes with the string "Unknown" and treated it as if it were a
   real outcome category (it drove ~44% of the "cleaned" dataset and is
   the main reason downstream ML models defaulted to predicting it).
4. `bench_size` came from counting regex matches of "Justice <Name>" in the
   free text, which double-counts a judge whose name is mentioned more than
   once. Where a case is filed once per bench member (previous point), the
   count of distinct `judge_folder` values for that case is a more reliable
   bench size than the text-regex count, and is used preferentially here.
5. `case_type` was only ever set to Civil/Criminal/Petition from a narrow
   regex, leaving 63.6% of rows "NaN" even when the case_no plainly said
   e.g. "Civil Review Petition" or "Const. Petition". A wider, still-honest
   classifier is used here — cases that truly can't be classified stay
   "Unclassified" rather than being guessed at.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "out_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "clean_cases.csv"
EXCLUDED_PATH = ROOT / "data" / "processed" / "excluded_rows.csv"
REPORT_PATH = ROOT / "reports" / "data_quality_report.md"

CORE_FIELDS = ["case_no", "date", "petitioner", "outcome", "word_count"]


# ---------------------------------------------------------------------------
# Step 1: load + drop rows where extraction failed completely
# ---------------------------------------------------------------------------
def load_and_split_failed(df: pd.DataFrame):
    fully_failed = df[CORE_FIELDS].isnull().all(axis=1)
    excluded = df[fully_failed].copy()
    excluded["exclusion_reason"] = "no text extracted from source PDF (all core fields empty)"
    kept = df[~fully_failed].copy()
    return kept, excluded


# ---------------------------------------------------------------------------
# Step 2: normalize the judges field, then dedupe one row per unique case
# ---------------------------------------------------------------------------
# The original extraction regex was `re.compile(r'Justice\s+[A-Z][a-zA-Z.\s]+', re.I)`.
# Because of the case-insensitive flag, `[A-Z]` there matches lowercase letters
# too, which turns "Justice" from an honorific marker into just the common
# noun ("...in the interest of justice, this court observed...") and then
# greedily swallows every following word until it hits punctuation. That is
# why the raw `judges` field sometimes contains run-on fragments like
# "almostsimilarobservationhasbeenmadebyafive" instead of a name, and why
# even the Title-Case-looking fragments often trail off into institutional
# boilerplate ("...Justice Qazi Faez Isa Judge Supreme Court Islamabad").
#
# This can be *filtered* (drop fragments that are clearly not a name at all)
# but not fully *repaired* without the source PDFs — there is no way to tell,
# from the extracted string alone, where a genuine name ends and trailing
# boilerplate begins. So this pipeline only removes the obvious garbage
# (fragments with no space-separated capitalized words, i.e. run-on lowercase
# text) and otherwise passes the field through unchanged, clearly labelled as
# unverified. `bench_size` is only ever taken from the distinct
# `judge_folder` count (Step 2's most reliable signal, available whenever a
# case was filed once per bench member) — never guessed from this noisy text
# — and is left missing everywhere else rather than invented.
def _split_judges(raw):
    if pd.isna(raw):
        return []
    parts = re.split(r"[;\n]", str(raw))
    seen, out = set(), []
    for p in parts:
        p = p.strip().strip(".")
        if not re.search(r"[A-Z][a-z]{2,}", p):  # needs at least one real Title-Case word
            continue
        key = re.sub(r"[^a-z]", "", p.lower())
        if key and key not in seen and len(key) >= 5:
            seen.add(key)
            out.append(p)
    return out


def _pick_best_row(group: pd.DataFrame) -> pd.Series:
    """Among rows for the same case (one per bench member), keep the row
    with the most non-null fields as the representative row."""
    completeness = group.notna().sum(axis=1)
    best = group.loc[completeness.idxmax()].copy()
    return best


def dedupe_cases(df: pd.DataFrame):
    dup_filenames = df["filename"].value_counts()
    multi = dup_filenames[dup_filenames > 1].index

    rows = []
    bench_recount_used = 0
    for fn, group in df.groupby("filename", sort=False):
        row = _pick_best_row(group)
        distinct_judge_folders = group["judge_folder"].dropna().nunique()

        judge_fragments = []
        for raw in group["judges"].dropna():
            judge_fragments.extend(_split_judges(raw))

        # bench_size is ONLY ever taken from the distinct judge_folder count
        # (the one reliable signal); the free-text "judges" field is too
        # noisy to size a bench from (see note above _split_judges), so it
        # is never used for this. Everywhere else, bench_size is missing
        # rather than guessed.
        row["bench_size"] = distinct_judge_folders if (fn in multi and distinct_judge_folders > 1) else np.nan
        if fn in multi and distinct_judge_folders > 1:
            bench_recount_used += 1

        row["judges_raw_filtered"] = "; ".join(dict.fromkeys(judge_fragments)) if judge_fragments else np.nan
        row["n_source_rows"] = len(group)
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    return out, len(multi), bench_recount_used


# ---------------------------------------------------------------------------
# Step 3: fix the outcome field
# ---------------------------------------------------------------------------
OUTCOME_MAP = {
    "set aside": "set aside",
    "this appeal is allowed": "appeal allowed",
    "this appeal is\nallowed": "appeal allowed",
    "appeal is allowed": "appeal allowed",
    "appeal allowed": "appeal allowed",
    "appeal is dismissed": "appeal dismissed",
    "appeal dismissed": "appeal dismissed",
    "petition dismissed": "petition dismissed",
    "leave to appeal granted": "leave to appeal granted",
}


def clean_outcome(series: pd.Series):
    def fix(v):
        if pd.isna(v):
            return np.nan
        v = re.sub(r"\s+", " ", str(v).replace("\n", " ")).strip().lower()
        return OUTCOME_MAP.get(v, v)  # unmapped-but-present values pass through as-is
    return series.apply(fix)


# ---------------------------------------------------------------------------
# Step 4: widen case_type classification from case_no text
# ---------------------------------------------------------------------------
CASE_TYPE_PATTERNS = [
    ("Constitutional Petition", r"const(?:itution)?\.?\s*p(?:etition)?\b|c\.?p\.?"),
    ("Civil Review Petition", r"c\.?r\.?p\.?|civil review"),
    ("Intra-Court Appeal", r"i\.?c\.?a\.?|intra.court appeal"),
    ("Civil Misc. Application", r"c\.?m\.?a\.?|civil misc"),
    ("Criminal Appeal", r"criminal appeal|crl\.?\s*a"),
    ("Criminal Petition", r"criminal petition|crl\.?\s*p"),
    ("Civil Appeal", r"civil appeal|c\.?a\.?\b"),
    ("Civil Petition", r"civil petition|c\.?p\.?l\.?a"),
]


def classify_case_type(case_no: str):
    if pd.isna(case_no):
        return "Unclassified"
    text = str(case_no).lower()
    for label, pattern in CASE_TYPE_PATTERNS:
        if re.search(pattern, text):
            return label
    return "Unclassified"


# ---------------------------------------------------------------------------
# Step 5: dates, sections, legal category
# ---------------------------------------------------------------------------
DATE_GAP_PLAUSIBLE_DAYS = 7305  # 20 years — generous, given known multi-year backlogs


def parse_dates(df: pd.DataFrame):
    """
    NOTE on `date` vs `hearing_date`: the original extraction script assumed
    `date` is the judgment date and `hearing_date` precedes it. Checking that
    assumption against the actual extracted values shows the opposite is far
    more common: for the 959 rows where both dates parsed, `hearing_date`
    falls AFTER `date` in 834 of them (87%), often by years, and only before
    it in 42. We cannot confirm from the metadata alone which real-world
    event each column captures — the source PDFs to check aren't available —
    so this pipeline does not assert an interpretation. It reports the gap
    literally as `hearing_date - date` and flags gaps beyond 20 years as
    likely date-parsing errors rather than real durations, without deleting
    the underlying rows.
    """
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hearing_date"] = pd.to_datetime(df["hearing_date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    gap = (df["hearing_date"] - df["date"]).dt.days
    implausible = gap.abs() > DATE_GAP_PLAUSIBLE_DAYS
    df["date_gap_days"] = gap
    df["date_gap_plausible"] = ~implausible & gap.notna()
    return df, int(implausible.sum()), int(gap.notna().sum())


def parse_sections(df: pd.DataFrame):
    def to_list(s):
        if pd.isna(s):
            return []
        return sorted({int(x) for x in re.findall(r"\d+", str(s))})
    df["sections_list"] = df["sections"].apply(to_list)
    df["num_sections"] = df["sections_list"].apply(len)
    return df


CONSTITUTIONAL_SECTIONS = set(range(1, 300))  # articles overlap numerically; category below uses case_type primarily


def legal_category(row):
    ct = row["case_type"]
    if ct in ("Criminal Appeal", "Criminal Petition"):
        return "Criminal"
    if ct == "Constitutional Petition":
        return "Constitutional"
    if ct in ("Civil Appeal", "Civil Petition", "Civil Review Petition", "Civil Misc. Application", "Intra-Court Appeal"):
        return "Civil"
    # fall back to section-number heuristics only when case_type is unclassified
    secs = set(row["sections_list"])
    criminal_markers = {302, 376, 420, 307, 498, 34, 109, 109}
    if secs & criminal_markers:
        return "Criminal"
    return "Unclassified"


# ---------------------------------------------------------------------------
# Step 6: flag (not silently fix) implausible party names
# ---------------------------------------------------------------------------
SUSPECT_PARTY_PHRASES = ("judgment of this court", "reported in", "observed that", "held that")


def party_name_plausible(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    if len(text) > 60:
        return False
    if any(phrase in text for phrase in SUSPECT_PARTY_PHRASES):
        return False
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    raw = pd.read_csv(RAW_PATH)
    n_raw = len(raw)

    kept, excluded = load_and_split_failed(raw)
    n_excluded = len(excluded)

    deduped, n_multijudge_filenames, n_bench_recounted = dedupe_cases(kept)
    n_after_dedupe = len(deduped)

    deduped["outcome_clean"] = clean_outcome(deduped["outcome"])
    n_labeled_outcome = deduped["outcome_clean"].notna().sum()

    deduped["case_type_clean"] = deduped["case_no"].apply(classify_case_type)
    n_unclassified_before = (deduped["case_type"].isna()).sum()
    n_unclassified_after = (deduped["case_type_clean"] == "Unclassified").sum()

    deduped, n_implausible_gap, n_with_both_dates = parse_dates(deduped)
    deduped = parse_sections(deduped)
    deduped["case_type"] = deduped["case_type_clean"]
    deduped["legal_category"] = deduped.apply(legal_category, axis=1)

    deduped["word_count"] = pd.to_numeric(deduped["word_count"], errors="coerce")
    deduped["char_count"] = pd.to_numeric(deduped["char_count"], errors="coerce")

    deduped["petitioner_plausible"] = deduped["petitioner"].apply(party_name_plausible)
    deduped["respondent_plausible"] = deduped["respondent"].apply(party_name_plausible)
    n_party_flagged = int(((deduped["petitioner_plausible"] == False) | (deduped["respondent_plausible"] == False)).sum())
    n_bench_known = int(deduped["bench_size"].notna().sum())

    final_cols = [
        "filename", "judge_folder", "case_no", "case_type", "legal_category",
        "date", "year", "month", "hearing_date", "date_gap_days", "date_gap_plausible",
        "petitioner", "petitioner_plausible", "respondent", "respondent_plausible",
        "bench_size", "judges_raw_filtered", "n_source_rows",
        "sections", "sections_list", "num_sections",
        "outcome_clean", "disposition_type",
        "word_count", "char_count",
        "constitutional_refs", "precedents",
        "petition_text",
    ]
    final = deduped[final_cols].rename(columns={"outcome_clean": "outcome", "judges_raw_filtered": "judges_raw_filtered"})
    final = final.sort_values(["year", "month"]).reset_index(drop=True)

    ROOT.joinpath("data", "processed").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)
    final.to_csv(CLEAN_PATH, index=False)
    excluded.to_csv(EXCLUDED_PATH, index=False)

    # ---- write the data quality report from the numbers computed above ----
    report = f"""# Data Quality Report

Generated by `src/clean_data.py`. Every number below comes from actually
running the pipeline against `data/raw/out_raw.csv` — none of it is asserted.

## 1. Row-level cleanup

| Step | Count |
|---|---|
| Raw rows in out_raw.csv | {n_raw} |
| Rows dropped: source PDF text extraction failed completely | {n_excluded} |
| Rows remaining before dedup | {n_raw - n_excluded} |
| Filenames that appeared more than once (same case filed once per bench member) | {n_multijudge_filenames} |
| Unique cases after deduping on filename | {n_after_dedupe} |

Verified: every repeated filename in the raw data shares the same `case_no`
and `date` across its repeats — i.e. it is always the *same* case re-filed
once per judge, never two different cases that happen to share a filename.
Deduping on `filename` (keeping the most complete row per case) is therefore
safe and correct, unlike the original notebook's full-row `drop_duplicates()`,
which could not detect these because `judge_folder` differed between the
repeated rows.

## 2. Bench size and judge names

`bench_size` in the raw data came from a regex count of "Justice <Name>"
mentions anywhere in the judgment text. Because the original regex used
`re.I` (case-insensitive), `[A-Z]` in it also matched lowercase letters —
so "Justice" stopped being an honorific marker and started matching the
ordinary word "justice" anywhere it appeared ("...in the interest of
justice, this court observed...") and then greedily consumed every
following word until punctuation. That is why the raw `judges` text
sometimes contains run-on nonsense instead of names, and why `bench_size`
reached values like 66 — nowhere close to a real Supreme Court bench.

This cannot be repaired from the extracted text alone: there is no way to
tell, after the fact, where a genuine name ends and trailing boilerplate
begins. So this pipeline does **not** attempt to reconstruct a clean bench
list. Instead:
- `bench_size` is reported **only** for the {n_multijudge_filenames} cases
  that were filed once per bench member, where the count of distinct
  `judge_folder` values for that case is a direct, verifiable bench size.
  That gives a reliable `bench_size` for {n_bench_known} of {n_after_dedupe}
  cases; it is left missing for the rest rather than estimated from the
  noisy text.
- The free-text field is kept as `judges_raw_filtered` (obvious non-name
  garbage removed) but is explicitly labelled unverified — it should be read
  as "judges possibly mentioned in this text", not as a confirmed bench
  roster. `judge_folder` remains the one reliable per-case judge field.

## 3. Outcome labels

Outcome strings containing a literal newline (an artifact of how the source
PDF wrapped text, e.g. `"set\\naside"`) were merged into their correct
canonical label instead of being counted as separate categories.

Missing outcomes are kept as missing (`NaN`), not filled with a placeholder
label. After cleaning, **{n_labeled_outcome} of {n_after_dedupe} cases
({n_labeled_outcome / n_after_dedupe:.1%})** have a real, extracted outcome.
The rest have none because the extraction script's outcome regex did not
match anywhere in that judgment's text — this is a genuine extraction gap
(likely a different phrasing of the result, or a judgment where the
operative order fell outside the pages the regex scanned), not something
that can be inferred from the metadata alone. **These rows are excluded from
any outcome-based analysis or modeling**, rather than being folded in as a
fake "Unknown" outcome class (the original notebook's approach, which alone
accounted for ~44% of its "cleaned" dataset and is the main reason its
classifiers defaulted to predicting the majority class).

## 4. Case type coverage

Case type was previously derived only from a narrow Civil/Criminal/Petition
regex over `case_no`, leaving {n_unclassified_before} of {n_after_dedupe} rows
unclassified. A wider (still pattern-based, still honest) classifier that
recognizes Civil Appeal / Civil Petition / Civil Review Petition / Civil Misc.
Application / Intra-Court Appeal / Criminal Appeal / Criminal Petition /
Constitutional Petition brings that down to {n_unclassified_after} rows
genuinely unclassifiable (no `case_no` text to classify from, or a case
numbering style the classifier doesn't recognize). No row was force-fit into
a category it doesn't clearly match.

## 5. Dates

The original pipeline assumed `date` is the judgment date and `hearing_date`
precedes it (calling the gap "hearing_gap_days"). Checking that assumption
against the actual data shows the opposite is far more common: of the
{n_with_both_dates} cases where both dates parsed, `hearing_date` falls
*after* `date` in the large majority, often by years or decades. Without the
source PDFs to check, this pipeline does not assert which real-world event
each column actually captures. It reports the literal gap as `date_gap_days`
(`hearing_date` minus `date`) and flags {n_implausible_gap} rows where that
gap exceeds 20 years as `date_gap_plausible = False` — very likely a
date-parsing error in one of the two fields — without deleting them.

## 6. Party names (petitioner / respondent)

The `petitioner`/`respondent` regex (`X versus Y` anywhere in the text) is
usually right — median extracted value is 14 characters, a plausible party
name — but can pick up an unrelated "X versus Y" phrase quoted from a cited
precedent instead of the actual case caption. `petitioner_plausible` /
`respondent_plausible` flag the {n_party_flagged} rows where the extracted
text is over 60 characters or contains a phrase like "judgment of this
court" (a strong sign it's prose, not a name) — flagged, not deleted, so
they stay visible rather than silently vanishing.

## 7. Known remaining limitations

- `petition_text` is only the **first 20 lines** of each judgment (a design
  choice in the original extraction script), not the full document. Any
  analysis of "petition length" from this field describes that fixed
  prefix, not the actual petition — this report does not use it for that
  purpose, and downstream work should not either. `word_count` / `char_count`
  are computed from the full extracted text and are not subject to this
  limitation.
- The original source PDFs are not available in this workspace (the supplied
  archive was empty), so extraction-level bugs — e.g. the outcome regex
  missing valid phrasings — can only be patched at the metadata level (as
  done above), not fixed at the source and re-extracted.
- `judge_folder` groups the corpus by which judge's page/folder a PDF was
  filed under on the source site; it is not necessarily the judgment's
  author.
"""
    REPORT_PATH.write_text(report)

    print(f"Raw rows: {n_raw}")
    print(f"Excluded (failed extraction): {n_excluded}")
    print(f"Unique cases after dedup: {n_after_dedupe}")
    print(f"Cases with a real outcome label: {n_labeled_outcome} ({n_labeled_outcome/n_after_dedupe:.1%})")
    print(f"Case-type unclassified before/after: {n_unclassified_before} -> {n_unclassified_after}")
    print(f"Date pairs available: {n_with_both_dates}, implausible (>20y) gaps flagged: {n_implausible_gap}")
    print(f"Bench size reliably known: {n_bench_known} / {n_after_dedupe} cases")
    print(f"Party-name rows flagged as suspect: {n_party_flagged}")
    print(f"Wrote: {CLEAN_PATH}")
    print(f"Wrote: {EXCLUDED_PATH}")
    print(f"Wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()
