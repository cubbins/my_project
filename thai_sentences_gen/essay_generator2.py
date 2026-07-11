

"""
Example 1 — Electronics paper
python essay_generator.py --field electronics --topic "Op-amp stability"


Example 2 — Economics paper
python essay_generator.py --field economics --topic "Monetary policy transmission"

Example 3 — Outline only
python essay_generator.py --field chemistry --topic "Catalyst poisoning" --outline_only

Example 4 — Using OpenAI backend
python essay_generator.py --field political_science --topic "Alliance formation" --backend openai --model gpt-4.1

python essay_generator.py --field electronics --topic "Op-amp stability" \
    --backend local \
    --model Qwen/Qwen2.5-3B-Instruct

python essay_generator.py --field electronics --topic "Op-amp stability" \
    --output_dir opamp_stability_run \
    --output opamp_stability.md


now run this:

python essay_generator.py --field electronics --topic "Op-amp stability" --output_dir essay_output

for 7B

python essay_generator.py --field electronics --topic "Op-amp stability" --model Qwen/Qwen2.5-7B-Instruct --output_dir essay_output

Qwen2.5‑3B‑Instruct

python essay_generator.py --field electronics --topic "Op-amp stability" --model Qwen/Qwen2.5-3B-Instruct --output_dir essay_output

python essay_generator2.py \
  --field literary_fiction \
  --topic "A pilgrim reflects on youth, knowledge, exile, and inward awakening" \
  --pdfs hesse_pdfs \
  --mode literary_influence \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir hesse_influenced_run \
  --output literary_essay.md




"""



import os
 
from datetime import datetime
import re
import glob
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LLM backends ---------------- #

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1"):
        if OpenAI is None:
            raise ImportError("openai package not installed.")
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content


class LocalTransformersBackend:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )

    def generate(self, system_prompt: str, user_prompt: str,
                 max_new_tokens: int = 800, temperature: float = 0.3):

        prompt = f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text


class LLMRouter:
    def __init__(self, backend: str = "openai", model: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIBackend(model=model or "gpt-4.1")
        elif backend == "local":
            if model is None:
                raise ValueError("Local backend requires model path.")
            self.llm = LocalTransformersBackend(model_path=model)
        else:
            raise ValueError("Unknown backend")

    def call(self, system_prompt: str, user_prompt: str) -> str:
        return self.llm.generate(system_prompt, user_prompt)


# ---------------- Data structures ---------------- #

@dataclass
class Chunk:
    id: str
    doc_id: str
    source_path: str
    page_num: int
    text: str
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class SectionPlan:
    title: str
    goal: str
    user_notes: str


@dataclass
class SectionDraft:
    title: str
    text: str
    citations: List[str]


# ---------------- PDF ingestion ---------------- #

def load_pdfs(pdf_folder: str) -> List[str]:
    return glob.glob(os.path.join(pdf_folder, "*.pdf"))


def extract_chunks_from_pdf(path: str,
                            doc_id: Optional[str] = None,
                            max_chars_per_chunk: int = 1200) -> List[Chunk]:
    doc = fitz.open(path)
    chunks: List[Chunk] = []
    doc_id = doc_id or str(uuid.uuid4())

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + max_chars_per_chunk, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        source_path=path,
                        page_num=page_num + 1,
                        text=chunk_text,
                    )
                )
            start = end

    return chunks


def build_corpus(pdf_folder: str) -> List[Chunk]:
    pdf_paths = load_pdfs(pdf_folder)
    corpus: List[Chunk] = []
    for path in pdf_paths:
        corpus.extend(extract_chunks_from_pdf(path))
    return corpus


# ---------------- Semantic index ---------------- #

class SemanticIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Chunk] = []
        self.emb_matrix: Optional[np.ndarray] = None

    def add_chunks(self, chunks: List[Chunk]):
        self.chunks.extend(chunks)

    def build(self):
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        for c, e in zip(self.chunks, embeddings):
            c.embedding = e
        self.emb_matrix = embeddings

    def search(self, query: str, top_k: int = 8) -> List[RetrievalResult]:
        if self.emb_matrix is None:
            raise RuntimeError("Index not built yet.")
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].reshape(1, -1)
        sims = cosine_similarity(q_emb, self.emb_matrix)[0]
        idxs = np.argsort(-sims)[:top_k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(sims[i]))
            for i in idxs
        ]


# ---------------- Essay generator core ---------------- #

class EssayGenerator22:
    def __init__(self,
                 llm: LLMRouter,
                 index: SemanticIndex,
                 field: str,
                 citation_style: str = "APA"):
        self.llm = llm
        self.index = index
        self.field = field
        self.citation_style = citation_style

    # ---- Outline ---- #

    def build_outline(self, topic: str, constraints: str = "") -> str:
        system_prompt = "You are an expert academic writing assistant."
        user_prompt = f"""
You are helping a student write a {self.field} paper.

Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a clear, logical outline (sections + subsections).
2. For each section, briefly state its purpose.
3. Keep the outline suitable for a 10–15 page term paper or research paper.
"""
        return self.llm.call(system_prompt, user_prompt)

    # ---- Section drafting ---- #

    def draft_section(self,
                      section_plan: SectionPlan,
                      retrieval_queries: List[str]) -> SectionDraft:
        # Aggregate retrieval results
        retrieved_chunks: List[RetrievalResult] = []
        for q in retrieval_queries:
            retrieved_chunks.extend(self.index.search(q, top_k=5))

        # Deduplicate by chunk id
        seen = set()
        unique_chunks: List[RetrievalResult] = []
        for r in retrieved_chunks:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                unique_chunks.append(r)

        context_blocks = []
        citation_labels = []
        for r in unique_chunks:
            c = r.chunk
            label = f"{os.path.basename(c.source_path)}, p.{c.page_num}"
            citation_labels.append(label)
            context_blocks.append(f"[{label}]\n{c.text}\n")

        context_text = "\n\n".join(context_blocks)

        system_prompt = "You are an academic writing assistant who grounds claims in provided sources."
        user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / arguments:
{section_plan.user_notes}

Retrieved reference context (from PDFs, with page hints):
{context_text}

Instructions:
- Write a coherent draft of this section.
- Use the user's notes as the backbone of the argument.
- Integrate reference material only when it clearly supports or contrasts the argument.
- When you use a specific passage, include an inline citation marker like [Source, p.X]; do NOT invent bibliographic details.
- Maintain an academic tone appropriate for {self.citation_style} style, but do NOT output the full reference list.
- Avoid hallucinating facts; if the context does not support a claim, say that evidence is limited.
"""

        text = self.llm.call(system_prompt, user_prompt)
        return SectionDraft(
            title=section_plan.title,
            text=text,
            citations=sorted(set(citation_labels))
        )

    # ---- Final assembly ---- #

    def assemble_essay(self,
                       title: str,
                       sections: List[SectionDraft]) -> str:
        parts = [f"# {title}\n"]
        for s in sections:
            parts.append(f"\n## {s.title}\n\n{s.text}\n")

        # Simple reference list from citation labels
        parts.append("\n## References (placeholders)\n")
        for c in sorted({cit for s in sections for cit in s.citations}):
            parts.append(f"- {c}")

        return "\n".join(parts)


# ---------------- Example CLI-like usage ---------------- #


# ---------------- Essay generator core ---------------- #

class EssayGenerator:
    def __init__(self,
                 llm: LLMRouter,
                 index: SemanticIndex,
                 field: str,
                 citation_style: str = "APA",
                 mode: str = "academic"):
        self.llm = llm
        self.index = index
        self.field = field
        self.citation_style = citation_style
        self.mode = mode

    # ---- Outline ---- #

    def build_outline(self, topic: str, constraints: str = "") -> str:
        if self.mode == "literary_influence":
            system_prompt = "You are a literary planning assistant."
            user_prompt = f"""
You are helping plan an original literary essay or prose work.

Field / genre: {self.field}
Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a clear literary structure with sections or movements.
2. For each section, describe its emotional, symbolic, and thematic purpose.
3. Emphasize inward conflict, symbolic development, narrative progression,
   philosophical reflection, and tonal continuity.
4. Do not copy or closely imitate any protected author.
"""
        else:
            system_prompt = "You are an expert academic writing assistant."
            user_prompt = f"""
You are helping a student write a {self.field} paper.

Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a clear, logical outline (sections + subsections).
2. For each section, briefly state its purpose.
3. Keep the outline suitable for a 10–15 page term paper or research paper.
"""
        return self.llm.call(system_prompt, user_prompt)

    # ---- Section drafting ---- #

    def draft_section(self,
                      section_plan: SectionPlan,
                      retrieval_queries: List[str]) -> SectionDraft:
        retrieved_chunks: List[RetrievalResult] = []

        for q in retrieval_queries:
            retrieved_chunks.extend(self.index.search(q, top_k=5))

        seen = set()
        unique_chunks: List[RetrievalResult] = []

        for r in retrieved_chunks:
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                unique_chunks.append(r)

        context_blocks = []
        citation_labels = []

        for r in unique_chunks:
            c = r.chunk
            label = f"{os.path.basename(c.source_path)}, p.{c.page_num}"
            citation_labels.append(label)
            context_blocks.append(f"[{label}]\n{c.text}\n")

        context_text = "\n\n".join(context_blocks)

        if self.mode == "literary_influence":
            system_prompt = """
You are a literary writing assistant.

Write original prose influenced only by broad, high-level literary traits
found in the provided reference passages.

Do not copy sentences, distinctive phrasing, unusual word sequences,
character names, scene structures, or protected expressive details.

Use the references only to infer general qualities such as:
- introspective narration
- symbolic journey structure
- spiritual or philosophical conflict
- nature imagery
- solitude and self-questioning
- gradual movement from confusion toward insight
- reflective pacing
- morally ambiguous inner development
"""
            user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / intended direction:
{section_plan.user_notes}

Reference context from PDFs:
{context_text}

Instructions:
- Write an original literary section on the topic.
- Capture broad thematic and tonal influence, not exact authorial imitation.
- Avoid direct copying, close paraphrase, or recognizable borrowed phrasing.
- Do not include academic citations inside the prose unless needed for commentary.
- Favor inward reflection, symbolic movement, philosophical tension,
  and carefully paced emotional development.
"""
        else:
            system_prompt = "You are an academic writing assistant who grounds claims in provided sources."
            user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / arguments:
{section_plan.user_notes}

Retrieved reference context (from PDFs, with page hints):
{context_text}

Instructions:
- Write a coherent draft of this section.
- Use the user's notes as the backbone of the argument.
- Integrate reference material only when it clearly supports or contrasts the argument.
- When you use a specific passage, include an inline citation marker like [Source, p.X]; do NOT invent bibliographic details.
- Maintain an academic tone appropriate for {self.citation_style} style, but do NOT output the full reference list.
- Avoid hallucinating facts; if the context does not support a claim, say that evidence is limited.
"""

        text = self.llm.call(system_prompt, user_prompt)

        return SectionDraft(
            title=section_plan.title,
            text=text,
            citations=sorted(set(citation_labels))
        )

    # ---- Final assembly ---- #

    def assemble_essay(self,
                       title: str,
                       sections: List[SectionDraft]) -> str:
        parts = [f"# {title}\n"]

        for s in sections:
            parts.append(f"\n## {s.title}\n\n{s.text}\n")

        if self.mode == "academic":
            parts.append("\n## References (placeholders)\n")
            for c in sorted({cit for s in sections for cit in s.citations}):
                parts.append(f"- {c}")

        else:
            parts.append("\n## Source Influence Notes\n")
            parts.append(
                "This draft used the supplied PDFs as high-level literary reference material "
                "for broad themes, tone, pacing, and symbolic structure. It is intended as "
                "original prose rather than a close imitation or reproduction."
            )

        return "\n".join(parts)


# ---------------- Example CLI-like usage ---------------- #


import argparse

def cli():
    parser = argparse.ArgumentParser(
        description="Full essay generator: outline → sections → citations → final draft"
    )

    parser.add_argument(
        "--field",
        type=str,
        required=True,
        help="Academic field (e.g., electronics, chemistry, economics, political_science)"
    )

    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="Essay topic (quoted string)"
    )

    parser.add_argument(
        "--pdfs",
        type=str,
        default="papers",
        help="Folder containing reference PDFs"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["openai", "local"],
        default="local",
        help="LLM backend to use"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="/models/Qwen2.5-7B-Instruct",
        help="Model name (OpenAI) or local model path (transformers)"
    )


    parser.add_argument(
        "--outline_only",
        action="store_true",
        help="Only generate outline and exit"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="final_essay.md",
        help="Filename for the final essay"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="essay_output",
        help="Directory to save essay output files"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["academic", "literary_influence"],
        default="academic",
        help="Writing mode"
    )


    args = parser.parse_args()



    # ---- Load PDFs + build index ----
    print(f"Loading PDFs from: {args.pdfs}")
    corpus = build_corpus(args.pdfs)
    print(f"Loaded {len(corpus)} chunks.")

    index = SemanticIndex()
    index.add_chunks(corpus)
    print("Building semantic index...")
    index.build()
    print("Index ready.")

    # ---- LLM backend ----
    llm_router = LLMRouter(
        backend=args.backend,
        model=args.model
    )

    generator = EssayGenerator(
        llm=llm_router,
        index=index,
        field=args.field,
        citation_style="APA"
    )

    # ---- Outline ----
    print("\n=== OUTLINE ===\n")
    outline = generator.build_outline(
        topic=args.topic,
        constraints="Include theory, empirical evidence, and implications."
    )
    print(outline)

    if args.outline_only:
        return

    # ---- Section plan generation (simple heuristic) ----
    # You can replace this with an LLM-based outline parser later.
    section_titles = [
        "Introduction",
        "Background & Theory",
        "Core Analysis",
        "Case Evidence",
        "Design / Implications",
        "Conclusion"
    ]

    section_plans = []
    for title in section_titles:
        section_plans.append(
            SectionPlan(
                title=title,
                goal=f"Explain the role of {args.topic} in {args.field}.",
                user_notes=f"""
My notes for this section on {title}:
- Focus on {args.topic}
- Connect to {args.field}
- Maintain academic tone
"""
            )
        )

    # ---- Draft sections ----
    sections = []
    for plan in section_plans:
        print(f"\n--- Drafting section: {plan.title} ---")
        queries = [
            args.topic,
            plan.title,
            args.field,
            "analysis",
            "technical details",
            "supply chain",
            "design",
        ]
        draft = generator.draft_section(plan, queries)
        sections.append(draft)

    # ---- Final essay ----
    final_essay = generator.assemble_essay(
        title=args.topic,
        sections=sections
    )


    print("\n=== FINAL ESSAY (Markdown) ===\n")
    print(final_essay)


    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save full essay
    with open(os.path.join(output_dir, args.output), "w") as f:
        f.write(final_essay)

    # Save each section safely
    for s in sections:
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', s.title)
        filename = f"{safe_title}.md"
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(s.text)


if __name__ == "__main__":
    cli()
