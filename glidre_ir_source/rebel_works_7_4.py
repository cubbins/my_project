from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load tokenizer and model (CPU-only)
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = (
    "The Loud Tour was the fourth overall and third world concert tour "
    "by Barbadian recording artist Rihanna."
)

# Prepare input
inputs = tokenizer(text, return_tensors="pt")

# Generate relation triples
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
        early_stopping=True
    )

# Decode output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\nExtracted Relations:")
print(output_text)

