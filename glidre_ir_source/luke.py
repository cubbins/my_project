import torch
from transformers import LukeTokenizer, LukeForEntityPairClassification

# ------------------------------------------------------------
# 1. Load LUKE tokenizer and model (CPU-only)
# ------------------------------------------------------------
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
model = LukeForEntityPairClassification.from_pretrained(
    "studio-ousia/luke-base",
    num_labels=5  # You can expand this
)

# Example IR text
text = (
    "The United States and Japan reaffirmed their security alliance "
    "during a meeting between President Biden and Prime Minister Kishida."
)

# ------------------------------------------------------------
# 2. Identify entity spans manually (IR actors/states)
# ------------------------------------------------------------
# In real use, you would run a NER model first.
entity_spans = [
    ("United States", 4, 17),
    ("Japan", 22, 27),
    ("President Biden", 63, 78),
    ("Prime Minister Kishida", 83, 105)
]

# ------------------------------------------------------------
# 3. Define IR relation labels
# ------------------------------------------------------------
id2label = {
    0: "ALLIANCE",
    1: "MEETING",
    2: "LEADER_OF",
    3: "STATE_ACTOR",
    4: "NO_RELATION"
}

# ------------------------------------------------------------
# 4. Run LUKE relation classification for each pair
# ------------------------------------------------------------
def classify_relation(text, ent1, ent2):
    name1, start1, end1 = ent1
    name2, start2, end2 = ent2

    inputs = tokenizer(
        text,
        entity_spans=[(start1, end1), (start2, end2)],
        entity_types=["ORG", "ORG"],  # treat states as ORG for LUKE
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    return id2label[pred]

# ------------------------------------------------------------
# 5. Evaluate all entity pairs
# ------------------------------------------------------------
print("\nExtracted IR Relations:")
for i in range(len(entity_spans)):
    for j in range(i + 1, len(entity_spans)):
        ent1 = entity_spans[i]
        ent2 = entity_spans[j]
        relation = classify_relation(text, ent1, ent2)
        print(f"{ent1[0]} — {ent2[0]}: {relation}")
