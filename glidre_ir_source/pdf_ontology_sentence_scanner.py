# pdf_ontology_sentence_scanner.py

import re
import csv
import json
from pathlib import Path

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize


PDF_FILE = r"Chouinard-affordances.pdf"
OUTPUT_CSV = "pdf_ontology_sentence_hits.csv"
OUTPUT_JSON = "pdf_ontology_sentence_hits.json"

# ------------------------------------------------------------
# 1. Ontological concept list
# ------------------------------------------------------------

ONTOLOGY = {
    "affordance": [
        "affordance",
        "afford",
        "affords",
        "afforded",
        "affording",
        "action possibility",
        "interaction possibility",
        "functional potential",
        "enabling property",
        "operational possibility"
    ],

    "artifact": [
        "artifact",
        "artefact",
        "object",
        "tool",
        "device",
        "technology",
        "interface",
        "system",
        "platform",
        "designed object"
    ],

    "mechanism": [
        "mechanism",
        "process",
        "operation",
        "causal process",
        "functional pathway",
        "operating principle"
    ],

    "perception": [
        "perception",
        "perceive",
        "perceived",
        "awareness",
        "recognition",
        "interpretation",
        "observation",
        "visibility"
    ],

    "dexterity": [
        "dexterity",
        "skill",
        "manual skill",
        "competence",
        "proficiency",
        "physical ability",
        "capability"
    ],

    "legitimacy": [
        "legitimacy",
        "legitimate",
        "authorization",
        "acceptance",
        "social approval",
        "institutional acceptance",
        "recognized validity"
    ],

    "constraint": [
        "constraint",
        "limitation",
        "restriction",
        "barrier",
        "inhibition",
        "structural limit"
    ]
}


# ------------------------------------------------------------
# 2. Setup
# ------------------------------------------------------------

def setup():
    nltk.download("punkt")


# ------------------------------------------------------------
# 3. Normalize text
# ------------------------------------------------------------

def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------------------------------------
# 4. Build searchable regex patterns
# ------------------------------------------------------------

def build_patterns(ontology):
    patterns = {}

    for root_term, terms in ontology.items():
        escaped_terms = []

        for term in terms:
            escaped = re.escape(term)
            escaped = escaped.replace(r"\ ", r"\s+")
            escaped_terms.append(escaped)

        pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
        patterns[root_term] = re.compile(pattern, re.IGNORECASE)

    return patterns


# ------------------------------------------------------------
# 5. Scan PDF page by page
# ------------------------------------------------------------

def scan_pdf(pdf_file, ontology):
    pdf_file = Path(pdf_file)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    patterns = build_patterns(ontology)
    results = []

    doc = fitz.open(pdf_file)

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        raw_text = page.get_text("text")
        page_text = clean_text(raw_text)

        if not page_text:
            continue

        sentences = sent_tokenize(page_text)

        for sentence_number, sentence in enumerate(sentences, start=1):
            for root_term, pattern in patterns.items():
                matches = pattern.findall(sentence)

                if matches:
                    results.append({
                        "pdf_file": pdf_file.name,
                        "page_number": page_number,
                        "sentence_number_on_page": sentence_number,
                        "root_concept": root_term,
                        "matched_terms": sorted(set(matches), key=str.lower),
                        "sentence": sentence
                    })

    doc.close()
    return results


# ------------------------------------------------------------
# 6. Save reports
# ------------------------------------------------------------

def save_csv(results, output_csv):
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "pdf_file",
            "page_number",
            "sentence_number_on_page",
            "root_concept",
            "matched_terms",
            "sentence"
        ])

        for r in results:
            writer.writerow([
                r["pdf_file"],
                r["page_number"],
                r["sentence_number_on_page"],
                r["root_concept"],
                "; ".join(r["matched_terms"]),
                r["sentence"]
            ])


def save_json(results, output_json):
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


# ------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------

def main():
    setup()

    results = scan_pdf(PDF_FILE, ONTOLOGY)

    save_csv(results, OUTPUT_CSV)
    save_json(results, OUTPUT_JSON)

    print(f"PDF scanned.")
    print(f"Matches found: {len(results)}")
    print(f"CSV report: {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")

    for r in results[:20]:
        print()
        print(f"Page {r['page_number']}, sentence {r['sentence_number_on_page']}")
        print(f"Concept: {r['root_concept']}")
        print(f"Matched: {', '.join(r['matched_terms'])}")
        print(r["sentence"])


if __name__ == "__main__":
    main()