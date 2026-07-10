from transformers import pipeline

# Choose a QA model: BERT or RoBERTa
MODEL_NAME = "deepset/bert-base-cased-squad2"       # BERT-QA
# MODEL_NAME = "deepset/roberta-base-squad2"        # RoBERTa-QA (uncomment to use)

# Initialize QA pipeline on GPU (device=0)
qa = pipeline(
    task="question-answering",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0  # Force GPU
)

# Context passage (you can replace this with enterprise text, PDFs, RAG output, etc.)
context = """
Photosynthesis is the process by which green plants convert light energy into chemical energy.
During photosynthesis, plants produce oxygen and glucose. The process occurs in the chloroplasts
and requires sunlight, water, and carbon dioxide.
"""

# Ask a question
question = "What do plants produce during photosynthesis?"

# Run QA
result = qa({
    "question": question,
    "context": context
})

print("\n=== Answer ===")
print(result)
