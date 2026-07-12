import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Babelscape/spanbert-base-cased-finetuned-ace"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "The United States and Japan reaffirmed their security alliance."

# ACE requires entity markers
marked = "The <e1>United States</e1> and <e2>Japan</e2> reaffirmed their security alliance."

inputs = tokenizer(marked, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1).item()

print("Predicted ACE relation:", model.config.id2label[pred])
