# ontology_synonym_builder.py

import json
import csv
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn


# ------------------------------------------------------------
# 1. One-time WordNet setup
# ------------------------------------------------------------

def setup_wordnet():
    nltk.download("wordnet")
    nltk.download("omw-1.4")


# ------------------------------------------------------------
# 2. Domain-specific technical vocabulary
# ------------------------------------------------------------

DOMAIN_ONTOLOGY = {
    "affordance": {
        "synonyms": [
            "action possibility",
            "interaction possibility",
            "functional potential",
            "enabling property",
            "operational possibility"
        ],
        "related_terms": [
            "constraint",
            "artifact",
            "user interaction",
            "design feature",
            "material environment"
        ],
        "broader_concept": "human-artifact interaction"
    },

    "artifact": {
        "synonyms": [
            "object",
            "tool",
            "device",
            "technology",
            "interface",
            "system",
            "designed object"
        ],
        "related_terms": [
            "material object",
            "technical system",
            "infrastructure",
            "platform"
        ],
        "broader_concept": "material object"
    },

    "mechanism": {
        "synonyms": [
            "process",
            "operation",
            "causal process",
            "functional pathway",
            "operating principle"
        ],
        "related_terms": [
            "cause",
            "effect",
            "condition",
            "procedure",
            "structure"
        ],
        "broader_concept": "causal explanation"
    },

    "legitimacy": {
        "synonyms": [
            "acceptance",
            "authorization",
            "recognized validity",
            "social approval",
            "institutional acceptance"
        ],
        "related_terms": [
            "authority",
            "norms",
            "institutions",
            "law",
            "custom"
        ],
        "broader_concept": "social recognition"
    },

    "perception": {
        "synonyms": [
            "awareness",
            "recognition",
            "interpretation",
            "observation",
            "apprehension"
        ],
        "related_terms": [
            "attention",
            "cognition",
            "sense-making",
            "visibility"
        ],
        "broader_concept": "cognition"
    },

    "dexterity": {
        "synonyms": [
            "skill",
            "manual skill",
            "competence",
            "proficiency",
            "physical ability"
        ],
        "related_terms": [
            "embodiment",
            "practice",
            "training",
            "capability"
        ],
        "broader_concept": "bodily capacity"
    },

    "constraint": {
        "synonyms": [
            "limitation",
            "restriction",
            "barrier",
            "inhibition",
            "structural limit"
        ],
        "related_terms": [
            "affordance",
            "refusal",
            "discouragement",
            "material condition"
        ],
        "broader_concept": "limiting condition"
    }
}


# ------------------------------------------------------------
# 3. WordNet synonym lookup
# ------------------------------------------------------------

def get_wordnet_synonyms(term):
    synonyms = set()

    for synset in wn.synsets(term):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != term.lower():
                synonyms.add(synonym)

    return sorted(synonyms)


# ------------------------------------------------------------
# 4. Build ontology entry for one term
# ------------------------------------------------------------

def build_entry(term):
    key = term.lower().strip()

    wordnet_synonyms = get_wordnet_synonyms(key)

    domain_data = DOMAIN_ONTOLOGY.get(key, {
        "synonyms": [],
        "related_terms": [],
        "broader_concept": ""
    })

    combined_synonyms = sorted(
        set(domain_data["synonyms"] + wordnet_synonyms)
    )

    entry = {
        "term": term,
        "canonical_term": key,
        "domain_synonyms": domain_data["synonyms"],
        "wordnet_synonyms": wordnet_synonyms,
        "combined_synonyms": combined_synonyms,
        "related_terms": domain_data["related_terms"],
        "broader_concept": domain_data["broader_concept"]
    }

    return entry


# ------------------------------------------------------------
# 5. Build ontology from list of terms
# ------------------------------------------------------------

def build_ontology(terms):
    return [build_entry(term) for term in terms]


# ------------------------------------------------------------
# 6. Save as JSON
# ------------------------------------------------------------

def save_json(ontology, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=4, ensure_ascii=False)


# ------------------------------------------------------------
# 7. Save as CSV
# ------------------------------------------------------------

def save_csv(ontology, output_file):
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "term",
            "canonical_term",
            "domain_synonyms",
            "wordnet_synonyms",
            "combined_synonyms",
            "related_terms",
            "broader_concept"
        ])

        for entry in ontology:
            writer.writerow([
                entry["term"],
                entry["canonical_term"],
                "; ".join(entry["domain_synonyms"]),
                "; ".join(entry["wordnet_synonyms"]),
                "; ".join(entry["combined_synonyms"]),
                "; ".join(entry["related_terms"]),
                entry["broader_concept"]
            ])


# ------------------------------------------------------------
# 8. Main program
# ------------------------------------------------------------

def main():
    setup_wordnet()

    input_terms = [
        "affordance",
        "artifact",
        "mechanism",
        "perception",
        "dexterity",
        "legitimacy",
        "constraint",
        "condition",
        "model",
        "structure"
    ]

    ontology = build_ontology(input_terms)

    save_json(ontology, "technical_ontology.json")
    save_csv(ontology, "technical_ontology.csv")

    for entry in ontology:
        print("\n" + entry["term"].upper())
        print("-" * len(entry["term"]))
        print("Combined synonyms:")
        for synonym in entry["combined_synonyms"]:
            print("  -", synonym)

        if entry["related_terms"]:
            print("Related terms:")
            for related in entry["related_terms"]:
                print("  -", related)

        if entry["broader_concept"]:
            print("Broader concept:", entry["broader_concept"])


if __name__ == "__main__":
    main()