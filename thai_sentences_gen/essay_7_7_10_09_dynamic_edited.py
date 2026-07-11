

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

python essay_generator2_corrected.py \
  --field literary_fiction \
  --topic "A pilgrim reflects on youth, knowledge, exile, and inward awakening" \
  --pdfs hesse_pdfs \
  --mode literary_influence \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir hesse_influenced_run \
  --output literary_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 500

  
python essay_generator2_corrected.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700

python essay_7_7_6_24AM.py \
  --field literary_fiction \
  --topic "style profile only" \
  --pdfs hesse_pdfs \
  --build_style_profile \
  --output_dir hesse_profile  

python essay_7_7_10_09_dynamic.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700

  python essay_7_7_10_09_dynamic_edited.py \
  --field literary_fiction \
  --topic "A man in debt wanders through the city while arguing with his own conscience" \
  --pdfs dostoevsky_pdfs \
  --mode literary_influence \
  --style_author dostoevsky \
  --backend local \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output_dir dostoevsky_influenced_run \
  --output psychological_essay.md \
  --top_k 2 \
  --max_context_chars 7000 \
  --max_new_tokens 700


"""



import os
 
from datetime import datetime
import re
import glob
import uuid
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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

        # Newer transformers prefers dtype over torch_dtype.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=self.device,
        )
        self.model.eval()

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 max_new_tokens: int = 600,
                 temperature: float = 0.4):
        """
        Generate only the assistant continuation, not the echoed prompt.
        Also uses inference_mode() to reduce GPU memory pressure.
        """
        prompt = (
            f"<system>\n{system_prompt}\n</system>\n"
            f"<user>\n{user_prompt}\n</user>\n"
            f"<assistant>\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000).to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode only newly generated tokens so the output does not include
        # the system/user prompt again.
        generated_ids = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if self.device == "cuda":
            torch.cuda.empty_cache()

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

    def call(self, system_prompt: str, user_prompt: str, **generate_kwargs) -> str:
        return self.llm.generate(system_prompt, user_prompt, **generate_kwargs)


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
    """Return PDF paths, including uppercase .PDF and PDFs in subdirectories."""
    patterns = [
        os.path.join(pdf_folder, "*.pdf"),
        os.path.join(pdf_folder, "*.PDF"),
        os.path.join(pdf_folder, "**", "*.pdf"),
        os.path.join(pdf_folder, "**", "*.PDF"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    return sorted(set(paths))

# ---------------- Document ingestion ---------------- #

def load_input_files(input_folder: str) -> List[str]:
    """
    Return supported input files from a folder, including subdirectories.
    Supports:
    - .pdf / .PDF
    - .txt / .TXT
    """
    patterns = [
        os.path.join(input_folder, "*.pdf"),
        os.path.join(input_folder, "*.PDF"),
        os.path.join(input_folder, "**", "*.pdf"),
        os.path.join(input_folder, "**", "*.PDF"),

        os.path.join(input_folder, "*.txt"),
        os.path.join(input_folder, "*.TXT"),
        os.path.join(input_folder, "**", "*.txt"),
        os.path.join(input_folder, "**", "*.TXT"),
    ]

    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    return sorted(set(paths))


def extract_chunks_from_txt(path: str,
                            doc_id: Optional[str] = None,
                            max_chars_per_chunk: int = 1200) -> List[Chunk]:
    chunks: List[Chunk] = []
    doc_id = doc_id or str(uuid.uuid4())

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()

    if not text:
        return chunks

    start = 0
    chunk_num = 1

    while start < len(text):
        end = min(start + max_chars_per_chunk, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    source_path=path,
                    page_num=chunk_num,   # for TXT, this means chunk number
                    text=chunk_text,
                )
            )

        start = end
        chunk_num += 1

    return chunks


def extract_chunks_from_document(path: str,
                                 doc_id: Optional[str] = None,
                                 max_chars_per_chunk: int = 1200) -> List[Chunk]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return extract_chunks_from_pdf(
            path=path,
            doc_id=doc_id,
            max_chars_per_chunk=max_chars_per_chunk
        )

    if ext == ".txt":
        return extract_chunks_from_txt(
            path=path,
            doc_id=doc_id,
            max_chars_per_chunk=max_chars_per_chunk
        )

    return []


def build_corpus(input_folder: str) -> List[Chunk]:
    input_paths = load_input_files(input_folder)

    corpus: List[Chunk] = []

    for path in input_paths:
        corpus.extend(extract_chunks_from_document(path))

    return corpus








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

class LegacyAcademicEssayGenerator:
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


# ---------------- Dynamic EssayGenerator loader ---------------- #

def load_essay_generator_class(module_path: Optional[str] = None):
    """
    Dynamically load EssayGenerator from a separate .py file on disk.

    By default, this loads essay_generator_class.py from the same directory
    as this primary program. This preserves the main pipeline while allowing
    EssayGenerator to be edited, generated, or replaced independently.
    """
    import importlib.util

    if module_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(here, "essay_generator_class.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"Could not find EssayGenerator module: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "dynamic_essay_generator_class",
        module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "EssayGenerator"):
        raise AttributeError(
            f"Module {module_path} does not define class EssayGenerator"
        )

    return module.EssayGenerator


EssayGenerator = load_essay_generator_class()

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
        help="Folder containing reference documents: .pdf, .PDF, .txt, or .TXT"
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

    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of retrieved chunks per query. Lower values reduce GPU memory use."
    )

    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=9000,
        help="Maximum retrieved context characters sent to the LLM per section."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=600,
        help="Maximum generated tokens per LLM call."
    )

    parser.add_argument(
        "--style_author",
        type=str,
        choices=["generic", "hesse", "dostoevsky"],
        default="generic",
        help="High-level literary style profile to condition generation"
    )



    args = parser.parse_args()



    # ---- Load PDFs + build index ----
    print(f"Loading PDFs from: {args.pdfs}")
    corpus = build_corpus(args.pdfs)
    print(f"Loaded {len(corpus)} chunks.")

    if not corpus:
        abs_pdf_dir = os.path.abspath(args.pdfs)
        raise FileNotFoundError(
            f"No extractable PDF text was found in: {abs_pdf_dir}\n"
            "Check that the directory exists, contains .pdf or .PDF files, "
            "and that the PDFs contain selectable text rather than scanned images."
        )

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
        citation_style="APA",
        mode=args.mode,
        style_author=args.style_author,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens
    )






    # ---- Outline ----
    print("\n=== OUTLINE ===\n")
    if args.mode == "literary_influence":
        outline_constraints = (
            "Create an original literary structure. Emphasize theme, tone, image-patterns, "
            "psychological development, and symbolic progression. Do not use academic term-paper framing."
        )
    else:
        outline_constraints = "Include theory, empirical evidence, and implications."

    outline = generator.build_outline(
        topic=args.topic,
        constraints=outline_constraints
    )
    print(outline)

    if args.outline_only:
        return

    # ---- Section plan generation (simple heuristic) ----
    # Later, this can be replaced with an LLM-based outline parser.
    if args.mode == "literary_influence":
        if args.style_author == "dostoevsky":
            section_titles = [
                "The First Agitation",
                "Pride and Poverty",
                "The Argument Within",
                "Humiliation",
                "Confession Without Relief",
                "Judgment and Unfinished Awakening",
            ]
            notes = f"""
My notes for this psychological movement:
- Focus on {args.topic}
- Use original prose, not academic explanation
- Emphasize guilt, pride, shame, debt, illness, social pressure, conscience, and unstable self-justification
- Prefer tense urban interiors and argumentative inward speech over serene nature imagery
- Do not imitate or copy any protected author directly
"""
        else:
            section_titles = [
                "Departure",
                "Youth and Restlessness",
                "Knowledge and Disillusionment",
                "Exile and Solitude",
                "Inward Awakening",
                "Return Without Arrival",
            ]
            notes = f"""
My notes for this literary movement:
- Focus on {args.topic}
- Use original prose, not academic explanation
- Emphasize inner conflict, symbolic landscape, memory, solitude, and gradual self-recognition
- Do not imitate or copy any protected author directly
"""

        section_plans = []
        for title in section_titles:
            section_plans.append(
                SectionPlan(
                    title=title,
                    goal=f"Develop the literary movement '{title}' within the topic: {args.topic}.",
                    user_notes=notes
                )
            )
    else:
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
        if args.mode == "literary_influence":
            if args.style_author == "dostoevsky":
                queries = [
                    args.topic,
                    plan.title,
                    "guilt pride shame poverty debt conscience confession fever illness city room",
                ]
            else:
                queries = [
                    args.topic,
                    plan.title,
                    "inward awakening solitude exile youth knowledge symbolic journey",
                ]
        else:
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
    with open(os.path.join(output_dir, args.output), "w", encoding="utf-8") as f:
        f.write(final_essay)

    # Save each section safely
    for s in sections:
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', s.title)
        filename = f"{safe_title}.md"
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(s.text)






if __name__ == "__main__":
    cli()
