# Auto-extracted EssayGenerator class
# This module is dynamically loaded by essay_7_7_10_09_dynamic.py.

from __future__ import annotations

import os
from typing import List

# These names are normally injected by the dynamic loader in the primary
# program. The fallback import allows this module to work when the primary
# script is executed as __main__.
try:
    SectionDraft
except NameError:
    try:
        from __main__ import SectionDraft, SectionPlan, RetrievalResult, LLMRouter, SemanticIndex
    except Exception:
        pass

class EssayGenerator:
    def __init__(self,
                 llm: LLMRouter,
                 index: SemanticIndex,
                 field: str,
                 citation_style: str = "APA",
                 mode: str = "academic",
                 style_author: str = "generic",
                 top_k: int = 2,
                 max_context_chars: int = 9000,
                 max_new_tokens: int = 600):
        self.llm = llm
        self.index = index
        self.field = field
        self.citation_style = citation_style
        self.mode = mode
        self.style_author = style_author
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens

    # ---- Outline ---- #

    def build_outline(self, topic: str, constraints: str = "") -> str:
        if self.mode == "literary_influence":
            system_prompt = "You are a literary planning assistant."

            if self.style_author == "dostoevsky":
                user_prompt = f"""
You are helping plan an original prose work influenced by broad, high-level
traits of nineteenth-century Russian psychological fiction.

Field / genre: {self.field}
Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a literary structure made of sections or movements.
2. Emphasize psychological crisis, moral contradiction, guilt, pride,
   humiliation, confession, feverish thought, spiritual anxiety, and social pressure.
3. Prefer cramped rooms, streets, stairs, debts, illness, overheard speech,
   accusations, conscience, unstable self-justification, and abrupt reversals.
4. Do not copy or closely imitate any protected author.
5. Do not use source character names, plot events, or recognizable scenes.
6. Do not frame the result as an academic term paper unless explicitly asked.
"""
            elif self.style_author == "hesse":
                user_prompt = f"""
You are helping plan an original literary essay or prose work influenced by
broad, high-level traits of spiritual and symbolic Bildungsroman prose.

Field / genre: {self.field}
Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a literary structure made of sections or movements.
2. For each section, describe its emotional, symbolic, and thematic purpose.
3. Emphasize inward conflict, symbolic development, solitude, apprenticeship,
   nature imagery, philosophical reflection, and tonal continuity.
4. Do not copy or closely imitate any protected author.
5. Do not use source character names, plot events, or recognizable scenes.
6. Do not frame the result as an academic term paper unless explicitly asked.
"""
            else:
                user_prompt = f"""
You are helping plan an original literary essay or prose work.

Field / genre: {self.field}
Topic: {topic}

Constraints:
{constraints}

Task:
1. Propose a literary structure made of sections or movements.
2. For each section, describe its emotional, symbolic, and thematic purpose.
3. Emphasize inward conflict, symbolic development, narrative progression,
   philosophical reflection, and tonal continuity.
4. Do not copy or closely imitate any protected author.
5. Do not frame the result as an academic term paper unless explicitly asked.
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
        return self.llm.call(system_prompt, user_prompt, max_new_tokens=self.max_new_tokens)

    # ---- Retrieval helpers ---- #

    def collect_context(self, retrieval_queries: List[str]) -> tuple[str, List[str]]:
        """
        Retrieve a small, deduplicated amount of context.
        This prevents CUDA out-of-memory errors caused by huge prompts.
        """
        retrieved_chunks: List[RetrievalResult] = []

        for q in retrieval_queries:
            retrieved_chunks.extend(self.index.search(q, top_k=self.top_k))

        seen = set()
        unique_chunks: List[RetrievalResult] = []

        for r in sorted(retrieved_chunks, key=lambda x: x.score, reverse=True):
            if r.chunk.id not in seen:
                seen.add(r.chunk.id)
                unique_chunks.append(r)

        context_blocks = []
        citation_labels = []
        total_chars = 0

        for r in unique_chunks:
            c = r.chunk
            label = f"{os.path.basename(c.source_path)}, p.{c.page_num}"
            block = f"[{label}; score={r.score:.3f}]\n{c.text.strip()}\n"

            if total_chars + len(block) > self.max_context_chars:
                break

            citation_labels.append(label)
            context_blocks.append(block)
            total_chars += len(block)

        return "\n\n".join(context_blocks), sorted(set(citation_labels))

    # ---- Section drafting ---- #

    def draft_section(self,
                      section_plan: SectionPlan,
                      retrieval_queries: List[str]) -> SectionDraft:
        context_text, citation_labels = self.collect_context(retrieval_queries)

        if self.mode == "literary_influence":
            if self.style_author == "dostoevsky":
                system_prompt = """
You are a literary writing assistant.

Write original prose influenced by broad, high-level traits of
nineteenth-century Russian psychological fiction.

Do not copy sentences, distinctive phrasing, unusual word sequences,
character names, scene structures, plot events, or protected expressive details.

Use the references only to infer general qualities such as:
- intense psychological interiority
- moral contradiction and self-accusation
- feverish, unstable reasoning
- shame, pride, humiliation, and resentment
- confessional narration
- abrupt emotional reversals
- religious and ethical anxiety
- long argumentative inner monologues
- conflict between intellect, guilt, ego, and conscience
- tense urban interiors rather than serene natural symbolism
- dialogue or thought that feels like interrogation, confession, or accusation
"""
                user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / intended direction:
{section_plan.user_notes}

Brief reference context from PDFs:
{context_text}

Instructions:
- Write an original literary section on the topic.
- Do not mention Dostoevsky or any source author inside the prose.
- Do not use source character names or recognizable plot events.
- Use a psychologically intense, morally conflicted, confessional tone.
- Let the narrator accuse himself, justify himself, contradict himself,
  and then doubt his own justification.
- Prefer cramped rooms, streets, debts, illness, shame, social unease,
  argument, conscience, and spiritual anxiety.
- Avoid calm river imagery, serene pilgrimage language, and generic
  “journey of self-discovery” phrasing unless the section specifically requires it.
- Avoid direct copying, close paraphrase, or recognizable borrowed phrasing.
- Do not include academic citations inside the prose unless needed for commentary.
"""
            elif self.style_author == "hesse":
                system_prompt = """
You are a literary writing assistant.

Write original prose influenced only by broad, high-level literary traits
found in the provided reference passages.

Do not copy sentences, distinctive phrasing, unusual word sequences,
character names, scene structures, plot events, or protected expressive details.

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

Brief reference context from PDFs:
{context_text}

Instructions:
- Write an original literary section on the topic.
- Capture broad thematic and tonal influence, not exact authorial imitation.
- Avoid direct copying, close paraphrase, source names, or recognizable borrowed phrasing.
- Do not include academic citations inside the prose unless needed for commentary.
- Favor inward reflection, symbolic movement, philosophical tension,
  and carefully paced emotional development.
"""
            else:
                system_prompt = """
You are a literary writing assistant.

Write original prose influenced only by broad, high-level literary traits
found in the provided reference passages.

Do not copy sentences, distinctive phrasing, unusual word sequences,
character names, scene structures, or protected expressive details.
"""
                user_prompt = f"""
Section title: {section_plan.title}
Section goal: {section_plan.goal}

User's own notes / intended direction:
{section_plan.user_notes}

Brief reference context from PDFs:
{context_text}

Instructions:
- Write an original literary section on the topic.
- Capture broad thematic and tonal influence, not exact authorial imitation.
- Avoid direct copying, close paraphrase, or recognizable borrowed phrasing.
- Do not include academic citations inside the prose unless needed for commentary.
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

        text = self.llm.call(system_prompt, user_prompt, max_new_tokens=self.max_new_tokens)

        return SectionDraft(
            title=section_plan.title,
            text=text,
            citations=citation_labels
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
            if self.style_author == "dostoevsky":
                parts.append(
                    "This draft used the supplied PDFs as high-level literary reference material "
                    "and was guided toward broad traits of nineteenth-century Russian psychological "
                    "fiction: moral conflict, confession, social pressure, guilt, pride, and "
                    "spiritual anxiety. It is intended as original prose rather than close imitation "
                    "or reproduction."
                )
            elif self.style_author == "hesse":
                parts.append(
                    "This draft used the supplied PDFs as high-level literary reference material "
                    "for broad themes, tone, pacing, symbolic development, and inward reflection. "
                    "It is intended as original prose rather than close imitation or reproduction."
                )
            else:
                parts.append(
                    "This draft used the supplied PDFs as high-level literary reference material "
                    "for broad themes, tone, pacing, and symbolic structure. It is intended as "
                    "original prose rather than a close imitation or reproduction."
                )

        return "\n".join(parts)
