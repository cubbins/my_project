# rebel_json_parser.py

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "Babelscape/rebel-large"


def parse_rebel_output(text: str):
    """
    Convert REBEL output like:

    <s><triplet> United States <subj> Japan <obj> diplomatic relation
    <triplet> Japan <subj> United States <obj> diplomatic relation</s>

    into JSON-style Python dictionaries.
    """

    text = (
        text.replace("<s>", "")
            .replace("</s>", "")
            .replace("<pad>", "")
            .strip()
    )

    triples = []
    current = {
        "subject": "",
        "object": "",
        "relation": ""
    }

    state = None

    tokens = text.split()

    for token in tokens:
        if token == "<triplet>":
            if current["subject"] and current["object"] and current["relation"]:
                triples.append(current)

            current = {
                "subject": "",
                "object": "",
                "relation": ""
            }
            state = "subject"

        elif token == "<subj>":
            state = "object"

        elif token == "<obj>":
            state = "relation"

        else:
            if state == "subject":
                current["subject"] += " " + token
            elif state == "object":
                current["object"] += " " + token
            elif state == "relation":
                current["relation"] += " " + token

    if current["subject"] and current["object"] and current["relation"]:
        triples.append(current)

    for triple in triples:
        triple["subject"] = triple["subject"].strip()
        triple["object"] = triple["object"].strip()
        triple["relation"] = triple["relation"].strip()

    return triples


def remove_symmetric_duplicates(triples):
    """
    Removes mirrored duplicate relations such as:

    United States -> Japan : diplomatic relation
    Japan -> United States : diplomatic relation

    Keeps only one copy.
    """

    seen = set()
    cleaned = []

    for t in triples:
        subject = t["subject"]
        object_ = t["object"]
        relation = t["relation"]

        key = tuple(sorted([subject, object_])) + (relation,)

        if key not in seen:
            seen.add(key)
            cleaned.append(t)

    return cleaned


def extract_relations(text: str, remove_duplicates: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    raw_output = tokenizer.decode(
        generated[0],
        skip_special_tokens=False
    )

    triples = parse_rebel_output(raw_output)

    if remove_duplicates:
        triples = remove_symmetric_duplicates(triples)

    result = {
        "source_text": text,
        "model": MODEL_NAME,
        "raw_model_output": raw_output,
        "relations": triples
    }

    return result


def main():
    text = "The United States and Japan reaffirmed their security alliance."

    result = extract_relations(text, remove_duplicates=True)

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
