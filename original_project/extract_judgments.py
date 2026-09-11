#!/usr/bin/env python3
"""
extract_judgments.py
Usage:
    python extract_judgments.py --input_dir "path/to/pdf_folder" --output_csv output.csv
"""

import os
import argparse
import re
import glob
import pdfplumber
import pandas as pd
from dateutil import parser as dateparser
from tqdm import tqdm

def extract_text_from_pdf(path):
    try:
        text_pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    text_pages.append(txt)
        full_text = "\n".join(text_pages).strip()
        return full_text
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return ""

# regex helpers
CASE_PATTERNS = [
    r'(Civil Appeal No\.[^\n,]*)',
    r'(Civil Petition No\.[^\n,]*)',
    r'(Criminal Petition No\.[^\n,]*)',
    r'(C\.A\. No\.[^\n,]*)',
    r'(Civil Appeal\s+[^\n]*)'
]

DATE_PATTERNS = [
    r'\b\d{1,2}\.\d{1,2}\.\d{4}\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b',
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
]

SECTION_PATTERN = re.compile(r'(?:Section|Sec|S|Article|Rule)\.?\s*(\d{1,4})', re.I)
PPC_NUMBER_PATTERN = re.compile(r'\b(\d{2,3})\b\s*(?:PPC|P\.P\.C\.|ppc)?', re.I)

OUTCOME_KEYWORDS = [
    r'\bappeal\s+is\s+allowed\b',
    r'\bthis\s+appeal\s+is\s+allowed\b',
    r'\bappeal\s+allowed\b',
    r'\bappeal\s+is\s+dismissed\b',
    r'\bappeal\s+dismissed\b',
    r'\bset\s+aside\b',
    r'\bleave\s+to\s+appeal\s+granted\b',
    r'\bpetition\s+dismissed\b'
]
OUTCOME_RE = re.compile("|".join(OUTCOME_KEYWORDS), re.I)

# new patterns
JUDGES_PATTERN = re.compile(r'Justice\s+[A-Z][a-zA-Z.\s]+', re.I)
ADV_PET_PATTERN = re.compile(r'For the Petitioner[s]?:\s*(.*)', re.I)
ADV_RES_PATTERN = re.compile(r'For the Respondent[s]?:\s*(.*)', re.I)
AOR_PATTERN = re.compile(r'AOR\s*:?(.+)', re.I)
ORIG_COURT_PATTERN = re.compile(r'from\s+the\s+judgment\s+of\s+(.+?)(?:,|\n)', re.I)
ORIG_CASE_PATTERN = re.compile(r'(RFA|W.P\.|Civil Misc|Suit No\.|Case No\.)[^\n,]*', re.I)
CONST_REF_PATTERN = re.compile(r'Article\s+\d+[A-Z]?', re.I)
PRECEDENT_PATTERN = re.compile(r'(PLD\s*\d{4}\s*SC\s*\d+|SCMR\s*\d{4}\s*\d+)', re.I)

def clean_party_name(name):
    if not name:
        return None
    return re.sub(r'\b(Petitioner|Respondent|Appellant|Defendant|Plaintiff)\b', '', name, flags=re.I).strip()

def find_first_match(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    return None

def extract_fields(text):
    # case number + type
    case_no = find_first_match(CASE_PATTERNS, text)
    case_type = None
    if case_no:
        if "Civil" in case_no: case_type = "Civil"
        elif "Criminal" in case_no: case_type = "Criminal"
        elif "Petition" in case_no: case_type = "Petition"

    # date
    found_date = None
    for dp in DATE_PATTERNS:
        m = re.search(dp, text, re.I)
        if m:
            s = m.group(0)
            try:
                dt = dateparser.parse(s, fuzzy=True)
                found_date = dt.date().isoformat()
                break
            except Exception:
                found_date = s
                break

    # hearing date
    hearing_date = None
    m = re.search(r'Date of hearing\s*:? ([^\n]+)', text, re.I)
    if m:
        try:
            hearing_date = dateparser.parse(m.group(1), fuzzy=True).date().isoformat()
        except Exception:
            hearing_date = m.group(1).strip()

    # petitioner vs respondent
    petitioner, respondent = None, None
    m = re.search(r'([^\n]{2,100})\s+versus\s+([^\n]{2,100})', text, re.I)
    if m:
        petitioner, respondent = clean_party_name(m.group(1)), clean_party_name(m.group(2))

    # advocates
    adv_pet = ADV_PET_PATTERN.findall(text)
    adv_res = ADV_RES_PATTERN.findall(text)
    aors = AOR_PATTERN.findall(text)

    advocates_petitioner = "; ".join(adv_pet) if adv_pet else None
    advocates_respondent = "; ".join(adv_res) if adv_res else None
    aors = "; ".join(aors) if aors else None

    # judges
    judges = JUDGES_PATTERN.findall(text)
    judges = "; ".join(set(judges)) if judges else None
    bench_size = len(set(judges.split("; "))) if judges else None
    judgment_author = judges.split(";")[-1].strip() if judges else None

    # originating court + case
    orig_court, orig_case = None, None
    m = ORIG_COURT_PATTERN.search(text)
    if m: orig_court = m.group(1).strip()
    m = ORIG_CASE_PATTERN.search(text)
    if m: orig_case = m.group(0).strip()

    # petition text (first 20 lines)
    petition_text = "\n".join(text.splitlines()[:20])

    # law sections
    sections = set()
    for m in SECTION_PATTERN.finditer(text):
        sections.add(m.group(1))
    for m in re.finditer(r'(\d{2,3})(?:\s*(?:PPC|ppc|P\.P\.C\.))', text):
        sections.add(m.group(1))
    sections_list = sorted(sections, key=lambda x:int(x)) if sections else []

    # outcome
    out_m = OUTCOME_RE.search(text)
    outcome = out_m.group(0).strip() if out_m else None

    # disposition + relief
    disposition_type, relief = None, None
    if outcome:
        if "dismissed" in outcome.lower():
            disposition_type = "Dismissed"
        elif "allowed" in outcome.lower() or "leave" in outcome.lower():
            disposition_type = "Allowed"
        elif "set aside" in outcome.lower():
            disposition_type = "Set Aside"
    m = re.search(r'(re-polling|re-hearing|notification.*set aside)', text, re.I)
    if m: relief = m.group(0).strip()

    # constitutional refs + precedents
    const_refs = "; ".join(set(CONST_REF_PATTERN.findall(text)))
    precedents = "; ".join(set(PRECEDENT_PATTERN.findall(text)))

    # basic stats
    word_count = len(text.split())
    char_count = len(text)

    return {
        "case_no": case_no,
        "case_type": case_type,
        "date": found_date,
        "hearing_date": hearing_date,
        "petitioner": petitioner,
        "respondent": respondent,
        "advocates_petitioner": advocates_petitioner,
        "advocates_respondent": advocates_respondent,
        "aors": aors,
        "judges": judges,
        "bench_size": bench_size,
        "judgment_author": judgment_author,
        "originating_court": orig_court,
        "originating_case_no": orig_case,
        "constitutional_refs": const_refs,
        "precedents": precedents,
        "petition_text": petition_text,
        "sections": ",".join(sections_list),
        "outcome": outcome,
        "disposition_type": disposition_type,
        "relief": relief,
        "word_count": word_count,
        "char_count": char_count
    }

def main(input_dir, output_csv, max_files=None):
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True))
    if not pdf_files:
        print("[ERROR] No PDF files found in", input_dir)
        return

    if max_files:
        pdf_files = pdf_files[:max_files]

    records = []
    empty_text_files = []

    for path in tqdm(pdf_files, desc="Processing PDFs"):
        text = extract_text_from_pdf(path)
        judge_folder = os.path.basename(os.path.dirname(path))  # parent folder = judge
        if not text or len(text.strip()) < 50:
            empty_text_files.append(path)
            print(f"[WARN] No text extracted from {os.path.basename(path)}")
            records.append({"filename": os.path.basename(path), "judge_folder": judge_folder})
            continue

        fields = extract_fields(text)
        rec = {"filename": os.path.basename(path), "judge_folder": judge_folder}
        rec.update(fields)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nDone. Wrote {len(df)} records to {output_csv}")
    if empty_text_files:
        print(f"Warning: {len(empty_text_files)} file(s) had no text extracted (likely scanned PDFs).")
        for f in empty_text_files[:10]:
            print(" -", f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing PDFs")
    ap.add_argument("--output_csv", default="extracted_judgments.csv", help="Output CSV file")
    ap.add_argument("--max_files", type=int, help="Limit how many PDFs to process (optional)")
    args = ap.parse_args()
    main(args.input_dir, args.output_csv, args.max_files)
