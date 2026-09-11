"""
Unit tests for the core cleaning logic in src/clean_data.py.

These test the specific bugs this project's audit fixed (see
reports/data_quality_report.md), not just happy paths — each one pins down
the behavior that made the original notebook's numbers wrong, so a future
change can't silently reintroduce them.

Run with: pytest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import (  # noqa: E402
    clean_outcome,
    classify_case_type,
    party_name_plausible,
    parse_dates,
    legal_category,
    _split_judges,
    DATE_GAP_PLAUSIBLE_DAYS,
)


# ---------------------------------------------------------------------------
# clean_outcome — the fix for outcomes split by a literal newline, and for
# treating missing outcomes as missing rather than a fake category.
# ---------------------------------------------------------------------------
def test_clean_outcome_merges_newline_wrapped_variant():
    result = clean_outcome(pd.Series(["set\naside", "set aside"]))
    assert result.tolist() == ["set aside", "set aside"]


def test_clean_outcome_leaves_missing_as_missing():
    result = clean_outcome(pd.Series([np.nan, "appeal allowed"]))
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == "appeal allowed"


def test_clean_outcome_normalizes_case_and_whitespace():
    result = clean_outcome(pd.Series(["  Appeal   Is  Allowed  "]))
    assert result.iloc[0] == "appeal allowed"


def test_clean_outcome_passes_through_unmapped_values():
    # An outcome phrasing the map doesn't recognize should survive as-is,
    # not get silently dropped or coerced to something else.
    result = clean_outcome(pd.Series(["remanded for fresh hearing"]))
    assert result.iloc[0] == "remanded for fresh hearing"


# ---------------------------------------------------------------------------
# classify_case_type — the widened classifier from data_quality_report.md §4.
# ---------------------------------------------------------------------------
def test_classify_case_type_recognizes_each_pattern():
    cases = {
        "Civil Appeal No.123 of 2020": "Civil Appeal",
        "Civil Petition No.45-K of 2019": "Civil Petition",
        "Criminal Appeal No.9 of 2021": "Criminal Appeal",
        "Criminal Petition No.7 of 2018": "Criminal Petition",
        "Const. Petition No.3 of 2022": "Constitutional Petition",
        "C.R.P. No.11 of 2017": "Civil Review Petition",
        "I.C.A. No.2 of 2020": "Intra-Court Appeal",
        "C.M.A. No.5 of 2020": "Civil Misc. Application",
    }
    for case_no, expected in cases.items():
        assert classify_case_type(case_no) == expected, case_no


def test_classify_case_type_unrecognized_text_is_unclassified_not_guessed():
    assert classify_case_type("Miscellaneous Reference No.1 of 2020") == "Unclassified"


def test_classify_case_type_missing_case_no_is_unclassified():
    assert classify_case_type(np.nan) == "Unclassified"


# ---------------------------------------------------------------------------
# party_name_plausible — flags suspect extractions instead of trusting them.
# ---------------------------------------------------------------------------
def test_party_name_plausible_accepts_short_name():
    assert party_name_plausible("Muhammad Ashraf") is True


def test_party_name_plausible_rejects_long_prose():
    assert party_name_plausible("x" * 61) is False


def test_party_name_plausible_rejects_suspect_phrase():
    assert party_name_plausible("as held in the judgment of this court") is False


def test_party_name_plausible_missing_stays_missing():
    assert pd.isna(party_name_plausible(np.nan))


# ---------------------------------------------------------------------------
# parse_dates — the date_gap_days / date_gap_plausible logic (§5). Reports
# the literal gap rather than asserting which column is the judgment date.
# ---------------------------------------------------------------------------
def test_parse_dates_computes_gap_and_flags_implausible_span():
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-01"],
        "hearing_date": ["2020-01-15", "1990-01-01"],  # 14 days vs ~30 years
    })
    out, n_implausible, n_both = parse_dates(df)
    assert n_both == 2
    assert n_implausible == 1
    assert out["date_gap_plausible"].tolist() == [True, False]
    assert out.loc[0, "date_gap_days"] == 14


def test_date_gap_plausible_threshold_is_twenty_years():
    assert DATE_GAP_PLAUSIBLE_DAYS == 7305


# ---------------------------------------------------------------------------
# legal_category — case_type takes priority; section-number heuristic is
# only a fallback for genuinely unclassified rows.
# ---------------------------------------------------------------------------
def test_legal_category_uses_case_type_first():
    row = pd.Series({"case_type": "Criminal Appeal", "sections_list": []})
    assert legal_category(row) == "Criminal"


def test_legal_category_falls_back_to_section_markers_when_unclassified():
    row = pd.Series({"case_type": "Unclassified", "sections_list": [302]})
    assert legal_category(row) == "Criminal"


def test_legal_category_unclassified_with_no_markers_stays_unclassified():
    row = pd.Series({"case_type": "Unclassified", "sections_list": [15]})
    assert legal_category(row) == "Unclassified"


# ---------------------------------------------------------------------------
# _split_judges — the fix for the re.I bug (§2) that let "Justice" match the
# ordinary word "justice" and swallow surrounding prose.
# ---------------------------------------------------------------------------
def test_split_judges_drops_lowercase_runon_garbage():
    garbage = "almostsimilarobservationhasbeenmadebyafivemember bench earlier"
    assert _split_judges(garbage) == []


def test_split_judges_keeps_title_case_fragment():
    result = _split_judges("Justice Qazi Faez Isa")
    assert result == ["Justice Qazi Faez Isa"]


def test_split_judges_dedupes_repeated_names():
    result = _split_judges("Justice Qazi Faez Isa\nJustice Qazi Faez Isa")
    assert len(result) == 1


def test_split_judges_missing_value_returns_empty_list():
    assert _split_judges(np.nan) == []
