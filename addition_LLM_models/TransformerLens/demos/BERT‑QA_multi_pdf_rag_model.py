import os
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
# Extract PDF text + metadata
# -------------------------------
def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = doc.metadata
    text_pages = [page.get_text("text") for page in doc]
    return text_pages, metadata

# -------------------------------
# Chunk text with page tracking
# -------------------------------
def chunk_pages(text_pages, max_chars=1200):
    chunks = []
    for page_num, text in enumerate(text_pages):
        sentences = sent_tokenize(text)
        current = ""
        for s in sentences:
            if len(current) + len(s) < max_chars:
                current += " " + s
            else:
                chunks.append((current.strip(), page_num))
                current = s
        if current:
            chunks.append((current.strip(), page_num))
    return chunks

# -------------------------------
# Build FAISS index with metadata
# -------------------------------
def build_multi_pdf_index(pdf_folder):
    all_chunks = []
    metadata_store = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text_pages, meta = extract_pdf(pdf_path)
            chunks = chunk_pages(text_pages)

            for chunk_text, page_num in chunks:
                metadata_store.append({
                    "pdf": filename,
                    "page": page_num,
                    "metadata": meta,
                    "chunk": chunk_text
                })
                all_chunks.append(chunk_text)

    embeddings = embedder.encode(all_chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, metadata_store

# -------------------------------
# Retrieve top-k chunks
# -------------------------------
def retrieve(question, index, metadata_store, k=5):
    q_emb = embedder.encode([question])
    distances, indices = index.search(q_emb, k)
    return [metadata_store[i] for i in indices[0]]

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
# Multi-PDF RAG QA
# -------------------------------
def rag_multi_pdf22(pdf_folder, question):
    index, metadata_store = build_multi_pdf_index(pdf_folder)
    retrieved = retrieve(question, index, metadata_store, k=5)

    best_answer, best_score = None, -1e9

    for item in retrieved:
        answer, score = answer_question(question, item["chunk"])
        if score > best_score and answer.strip():
            best_score = score
            best_answer = {
                "answer": answer,
                "score": score,
                "pdf": item["pdf"],
                "page": item["page"],
                "metadata": item["metadata"],
                "context_excerpt": item["chunk"][:300] + "..."
            }

    return best_answer

def rag_multi_pdf(pdf_folder, question):
    index, metadata_store = build_multi_pdf_index(pdf_folder)
    retrieved = retrieve(question, index, metadata_store, k=50)

    answers_by_pdf = {}

    for item in retrieved:
        pdf = item["pdf"]
        answer, score = answer_question(question, item["chunk"])

        if answer.strip():
            if pdf not in answers_by_pdf:
                answers_by_pdf[pdf] = []

            answers_by_pdf[pdf].append({
                "answer": answer,
                "score": score,
                "page": item["page"],
                "metadata": item["metadata"],
                "context_excerpt": item["chunk"][:300] + "..."
            })

    # Sort answers inside each PDF by score
    for pdf in answers_by_pdf:
        answers_by_pdf[pdf] = sorted(
            answers_by_pdf[pdf],
            key=lambda x: x["score"],
            reverse=True
        )

    # Write report to file
    write_rag_report(answers_by_pdf, "rag_report.txt")

    return answers_by_pdf


def write_rag_report(answers_by_pdf, output_path="rag_report.txt"):
    """
    Create a paragraph-by-paragraph report of grouped RAG answers,
    followed by a bibliography-style section listing all PDFs used.
    """

    lines = []
    lines.append("RAG Multi-PDF Answer Report\n")
    lines.append("=" * 60 + "\n\n")

    # Paragraph-by-paragraph section
    for pdf_name, answers in answers_by_pdf.items():
        lines.append(f"Document: {pdf_name}\n")
        lines.append("-" * 60 + "\n")

        for idx, ans in enumerate(answers, start=1):
            lines.append(f"Paragraph {idx}:\n")
            lines.append(f"  Answer: {ans['answer']}\n")
            lines.append(f"  Score: {ans['score']:.4f}\n")
            lines.append(f"  Page: {ans['page']}\n")
            lines.append(f"  Excerpt: {ans['context_excerpt']}\n")
            lines.append("\n")

        lines.append("\n")

    # Bibliography section
    lines.append("\nBibliography\n")
    lines.append("=" * 60 + "\n")

    for pdf_name, answers in answers_by_pdf.items():
        meta = answers[0]["metadata"] if answers else {}
        title = meta.get("title", "Unknown Title")
        author = meta.get("author", "Unknown Author")
        creation = meta.get("creationDate", "Unknown Date")

        lines.append(f"- {pdf_name}\n")
        lines.append(f"    Title: {title}\n")
        lines.append(f"    Author: {author}\n")
        lines.append(f"    Created: {creation}\n")
        lines.append("\n")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return output_path



# -------------------------------
# Test
# -------------------------------
if __name__ == "__main__":
    pdf_folder = "pdfs/"   # folder containing multiple PDFs
    question = "What conditions shape affordances?"
    print("the question in this multi RAG model is - ",question)
    result = rag_multi_pdf(pdf_folder, question)
    print(json.dumps(result, indent=2))
    print("producing rag_report.txt")

