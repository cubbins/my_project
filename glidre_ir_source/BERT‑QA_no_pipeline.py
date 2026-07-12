import re
import json
from pathlib import Path

import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

nltk.download("punkt")

MODEL_NAME = "deepset/bert-base-cased-squad2"

# Load model + tokenizer manually (no pipeline)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to("cuda")

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            all_text.append(text)

    return "\n".join(all_text)

def chunk_text(text, max_chars=1200):
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

def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores) + 1

    answer_ids = inputs["input_ids"][0][start:end]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    score = float(start_scores[0][start] + end_scores[0][end - 1])

    return answer, score

def answer_question_over_pdf(pdf_path, question):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)

    best_answer = None
    best_score = -1e9

    for idx, chunk in enumerate(chunks):
        answer, score = answer_question(question, chunk)

        if score > best_score and answer.strip():
            best_score = score
            best_answer = {
                "answer": answer,
                "score": score,
                "chunk_index": idx,
                "context_excerpt": chunk[:300] + "..."
            }

    return best_answer

if __name__ == "__main__":
    pdf_path = "example.pdf"
    pdf_path = "Chouinard-affordances.pdf"
    question = "What is the main economic argument presented?"

    result = answer_question_over_pdf(pdf_path, question)
    print(json.dumps(result, indent=2))
