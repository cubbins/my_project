import re
import json
import fitz
import nltk
import torch
import faiss
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer

nltk.download("punkt")

# -------------------------------
# Load QA model (GPU)
# -------------------------------
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2").to("cuda")

# -------------------------------
# Load embedding model (CPU)
# -------------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------------
# PDF extraction
# -------------------------------
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text("text") for page in doc)

# -------------------------------
# Chunking
# -------------------------------
def chunk_text(text, max_chars=1200):
    sentences = sent_tokenize(text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

# -------------------------------
# Build FAISS index
# -------------------------------
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# -------------------------------
# Retrieve top-k chunks
# -------------------------------
def retrieve_chunks(question, chunks, index, k=5):
    q_emb = embedder.encode([question])
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

# -------------------------------
# Manual QA inference
# -------------------------------
def answer_question(question, context):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer_ids = inputs["input_ids"][0][start:end]
    answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True)
    score = float(outputs.start_logits[0][start] + outputs.end_logits[0][end - 1])
    return answer, score

# -------------------------------
# RAG QA over PDF
# -------------------------------
def rag_answer(pdf_path, question):
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    index, _ = build_faiss_index(chunks)
    retrieved = retrieve_chunks(question, chunks, index, k=5)

    best_answer, best_score = None, -1e9
    for chunk in retrieved:
        answer, score = answer_question(question, chunk)
        if score > best_score and answer.strip():
            best_score = score
            best_answer = {
                "answer": answer,
                "score": score,
                "context_excerpt": chunk[:300] + "..."
            }
    return best_answer

# -------------------------------
# Test
# -------------------------------
if __name__ == "__main__":
    pdf_path = "Chouinard-affordances.pdf"
    question = "What conditions shape affordances?"
    print("using the rag model - question",question)
    result = rag_answer(pdf_path, question)
    print(json.dumps(result, indent=2))
