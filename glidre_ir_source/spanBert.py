import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------
# 1. Load SpanBERT ACE model (CPU-only)
# ------------------------------------------------------------
model_name = "mrm8488/spanbert-finetuned-ace2005-relation-extraction"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ------------------------------------------------------------
# 2. Example IR text
# ------------------------------------------------------------
text = (
    "The United States and Japan reaffirmed their security alliance "
    "during a meeting between President Biden and Prime Minister Kishida."
)

# ACE requires marking entity spans with special tokens
marked_text = (
    "The <e1>United States</e1> and <e2>Japan</e2> reaffirmed their security alliance."
)

# ------------------------------------------------------------
# 3. Tokenize
# ------------------------------------------------------------
inputs = tokenizer(marked_text, return_tensors="pt")

# ------------------------------------------------------------
# 4. Forward pass (CPU-only)
# ------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()

# ------------------------------------------------------------
# 5. ACE label mapping
# ------------------------------------------------------------
id2label = model.config.id2label

print("\nPredicted ACE Relation:")
print(id2label[pred_id])
