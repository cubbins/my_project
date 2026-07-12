import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, XLMRobertaModel

# ------------------------------------------------------------
# 1. Load tokenizer from HuggingFace
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("cea-list-ia/glidre_multi")

# ------------------------------------------------------------
# 2. Load GLiDRE model weights manually
# ------------------------------------------------------------
# state_dict = torch.load("glidre_multi/pytorch_model.bin", map_location="cpu")

state_dict = torch.load(
    "glidre_multi/pytorch_model.bin",
    map_location="cpu",
    weights_only=False
)



# Inspect keys if needed
# print(state_dict.keys())

# ------------------------------------------------------------
# 3. Load encoder (XLM-Roberta base)
# ------------------------------------------------------------
encoder = XLMRobertaModel.from_pretrained("xlm-roberta-base")

# Load encoder weights from GLiDRE
encoder_state = {}
for k, v in state_dict.items():
    if k.startswith("encoder."):
        new_key = k.replace("encoder.", "")
        encoder_state[new_key] = v

encoder.load_state_dict(encoder_state, strict=False)

# ------------------------------------------------------------
# 4. Build classifier head
# ------------------------------------------------------------
num_labels = 3  # COUNTRY_OF_CITIZENSHIP, PUBLICATION_DATE, PART_OF
classifier = nn.Linear(encoder.config.hidden_size, num_labels)

classifier.load_state_dict({
    "weight": state_dict["classifier.weight"],
    "bias": state_dict["classifier.bias"]
})

# ------------------------------------------------------------
# 5. Input text
# ------------------------------------------------------------
text = (
    "The Loud Tour was the fourth overall and third world concert tour "
    "by Barbadian recording artist Rihanna."
)

labels = [
    "COUNTRY_OF_CITIZENSHIP",
    "PUBLICATION_DATE",
    "PART_OF"
]

# ------------------------------------------------------------
# 6. Tokenize
# ------------------------------------------------------------
inputs = tokenizer(text, return_tensors="pt")

# ------------------------------------------------------------
# 7. Forward pass (CPU-only)
# ------------------------------------------------------------
with torch.no_grad():
    outputs = encoder(**inputs)
    pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
    logits = classifier(pooled)
    probs = F.softmax(logits, dim=-1).squeeze()

# ------------------------------------------------------------
# 8. Print results
# ------------------------------------------------------------
print("\nPredicted Relations:")
for label, p in zip(labels, probs):
    print(f"{label}: {p.item():.4f}")
