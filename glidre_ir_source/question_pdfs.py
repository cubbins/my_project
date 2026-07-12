

# available tasks are ['any-to-any', 'audio-classification', 'automatic-speech-recognition', 
# 'depth-estimation', 'document-question-answering', 'feature-extraction', 
# 'fill-mask', 'image-classification', 'image-feature-extraction', 
# 'image-segmentation', 'image-text-to-text', 'keypoint-matching', 
# 'mask-generation', 'ner', 'object-detection', 'sentiment-analysis', 
# 'table-question-answering', 'text-classification', 'text-generation', 
# 'text-to-audio', 'text-to-speech', 'token-classification', 
# 'video-classification', 'zero-shot-audio-classification', 'zero-shot-classification', 
# 'zero-shot-image-classification', 'zero-shot-object-detection']"





import re
import json
from pathlib import Path

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

from transformers import pipeline

# Make sure NLTK has the tokenizer
nltk.download("punkt")

# -------------------------------
# 1. Load QA model (GPU)
# -------------------------------
MODEL_NAME = "deepset/bert-base-cased-squad2"

qa = pipeline(
    task="text-question-answering",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0
)



# -------------------------------
# 2. Extract text from PDF
# -------------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            all_text.append(text)

    return "\n".join(all_text)

# -------------------------------
# 3. Chunk text into passages
# -------------------------------
def chunk_text(text, max_chars=1200):
    """
    Break text into chunks that fit QA model limits.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks

# -------------------------------
# 4. Run QA over all chunks
# -------------------------------
def answer_question_over_pdf(pdf_path, question):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)

    best_answer = None
    best_score = 0.0

    for idx, chunk in enumerate(chunks):
        result = qa({"question": question, "context": chunk})

        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = {
                "answer": result["answer"],
                "score": result["score"],
                "chunk_index": idx,
                "context_excerpt": chunk[:300] + "..."
            }

    return best_answer

# -------------------------------
# 5. Test the system
# -------------------------------
if __name__ == "__main__":
    pdf_path = "example.pdf"   # <-- your PDF file
    # Chouinard-affordances.pdf
    pdf_path = "Chouinard-affordances.pdf"   # <-- your PDF file   
    question = "What is the main economic argument presented?"

    result = answer_question_over_pdf(pdf_path, question)

    print(json.dumps(result, indent=2))
