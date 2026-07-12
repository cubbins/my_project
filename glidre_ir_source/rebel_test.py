import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Babelscape/rebel-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "The United States and Japan reaffirmed their security alliance."

inputs = tokenizer(text, return_tensors="pt", truncation=True)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
    )

output = tokenizer.decode(generated[0], skip_special_tokens=False)

print(output)
