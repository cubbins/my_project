import json
import torch
from gliner import GLiNER
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


TEXT = """
China announced additional export controls on gallium and germanium.
The United States criticized the decision.
NATO members discussed supply chain resilience during a meeting in Brussels.
"""


ENTITY_LABELS = [
    "Country",
    "Organization",
    "Alliance",
    "Strategic Resource",
    "Technology",
    "Export Control",
    "Economic Sanction",
    "Conflict",
    "Location",
]


def run_gliner(text: str):
    print("=" * 80)
    print("Loading GLiNER...")
    print("=" * 80)

    model = GLiNER.from_pretrained("urchade/gliner_large-v2")

    print("Running GLiNER entity extraction...")
    entities = model.predict_entities(text, ENTITY_LABELS, threshold=0.35)

    cleaned = []
    for ent in entities:
        cleaned.append({
            "text": ent.get("text"),
            "label": ent.get("label"),
            "score": float(ent.get("score", 0.0)),
            "start": ent.get("start"),
            "end": ent.get("end"),
        })

    return cleaned


def extract_rebel_triplets(generated_text: str):
    triplets = []
    relation = ""
    subject = ""
    object_ = ""
    current = None

    tokens = (
        generated_text
        .replace("<s>", "")
        .replace("<pad>", "")
        .replace("</s>", "")
        .split()
    )

    for token in tokens:
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append({
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip(),
                })
            subject = ""
            relation = ""
            object_ = ""
            current = "subject"
        elif token == "<subj>":
            current = "object"
        elif token == "<obj>":
            current = "relation"
        else:
            if current == "subject":
                subject += " " + token
            elif current == "object":
                object_ += " " + token
            elif current == "relation":
                relation += " " + token

    if subject and relation and object_:
        triplets.append({
            "subject": subject.strip(),
            "relation": relation.strip(),
            "object": object_.strip(),
        })

    return triplets


def run_rebel(text: str):
    print("=" * 80)
    print("Loading REBEL...")
    print("=" * 80)

    model_name = "Babelscape/rebel-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print("Running REBEL relation extraction...")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]
    triplets = extract_rebel_triplets(decoded)

    return decoded, triplets


def main():
    print("\nINPUT TEXT:")
    print(TEXT)

    entities = run_gliner(TEXT)
    rebel_raw, triplets = run_rebel(TEXT)

    result = {
        "source_text": TEXT.strip(),
        "entities": entities,
        "relations": triplets,
        "rebel_raw_output": rebel_raw,
    }

    print("\n" + "=" * 80)
    print("FINAL JSON RESULT")
    print("=" * 80)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open("gliner_rebel_unit_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nSaved: gliner_rebel_unit_result.json")


if __name__ == "__main__":
    main()