from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer and model from HuggingFace
tokenizer = AutoTokenizer.from_pretrained("cea-list-ia/glidre_multi")
model = AutoModelForSequenceClassification.from_pretrained("cea-list-ia/glidre_multi")

text = "The Loud Tour was the fourth overall and third world concert tour by Barbadian recording artist Rihanna."

# Your entity mentions
mentions = [
    {"id": 0, "mentions": [{"value": "Barbadian", "start": 69, "end": 78}], "type": "LOC"},
    {"id": 1, "mentions": [{"value": "Rihanna", "start": 96, "end": 103}], "type": "PER"}
]

# Relation labels you want to test
labels = ["COUNTRY_OF_CITIZENSHIP", "PUBLICATION_DATE", "PART_OF"]

# Encode the text
inputs = tokenizer(text, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: [batch, num_labels]

# Convert logits to probabilities
probs = F.softmax(logits, dim=-1).squeeze()

# Print results
print("Predicted Relations:")
for i, label in enumerate(labels):
    print(f"{label}: {probs[i].item():.4f}")
