import json
import re
import ast
from typing import List

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer
from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

RESEARCH_QUESTION = "Are liberal democracies portrayed as more vulnerable to AI competition?"

JSON_OUTPUT_FILE = "computed_research_result.json"
MARKDOWN_OUTPUT_FILE = "computed_research_report.md"

WIKIPEDIA_SITE_FILTER = "site:en.wikipedia.org/wiki"
MAX_SEARCH_ANGLES = 6
MAX_WIKIPEDIA_RESULTS_PER_ANGLE = 8
MAX_PAGES_IN_SYNTHESIS = 20


class WebPageResult(BaseModel):
    title: str
    url: str
    reason: str


class ResearchReport(BaseModel):
    topic: str
    search_angles: List[str]
    wikipedia_pages: List[WebPageResult]
    synthesis: str


def load_agent(max_steps: int = 2) -> CodeAgent:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TransformersModel(
        model_id=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return CodeAgent(
        tools=[],
        model=model,
        add_base_tools=False,
        max_steps=max_steps,
    )


def parse_list_from_agent(raw_text: str) -> List[str]:
    text = str(raw_text).strip()

    # First try strict Python list parsing
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Fallback: line-based parsing with bullet/number stripping
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        # Remove leading bullets, numbers, etc.
        line = re.sub(r"^[-*\d.)\s]+", "", line).strip()
        if line:
            lines.append(line)

    return lines


def compute_search_angles(agent: CodeAgent, question: str) -> List[str]:
    prompt = f"""
Return only a Python list of 5 search queries.

Research question:
{question}

The search queries should help locate Wikipedia pages related to:
- liberal democracy
- AI competition
- authoritarianism
- disinformation
- AI governance
- surveillance
- democratic vulnerability

Return only the list. No explanation.
"""

    raw = agent.run(prompt)
    angles = parse_list_from_agent(str(raw))

    # Fallback if the agent output is too weak or malformed
    if len(angles) < 3:
        angles = [
            "liberal democracy AI competition",
            "artificial intelligence authoritarianism democracy",
            "algorithmic disinformation democracy",
            "AI governance democratic states",
            "surveillance capitalism democracy",
            "computational propaganda democracy",
        ]

    return angles[:MAX_SEARCH_ANGLES]


def extract_wikipedia_urls(text: str) -> List[str]:
    urls = re.findall(
        r"https?://(?:en\.)?(?:m\.)?wikipedia\.org/wiki/[^\s\]\)\"']+",
        text,
    )

    cleaned_urls: List[str] = []
    for url in urls:
        # Normalize mobile URLs
        url = url.replace("https://en.m.wikipedia.org", "https://en.wikipedia.org")
        url = url.replace("http://en.m.wikipedia.org", "https://en.wikipedia.org")

        # Remove fragment identifiers
        url = url.split("#")[0]

        if "/wiki/" not in url:
            continue

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    return cleaned_urls


def wikipedia_search(
    search_tool: DuckDuckGoSearchTool,
    query: str,
    max_results: int = MAX_WIKIPEDIA_RESULTS_PER_ANGLE,
) -> List[WebPageResult]:
    raw = search_tool.forward(f"{WIKIPEDIA_SITE_FILTER} {query}")
    text = str(raw)

    urls = extract_wikipedia_urls(text)

    results: List[WebPageResult] = []
    for url in urls[:max_results]:
        title = url.rsplit("/", 1)[-1].replace("_", " ")

        # Filter out malformed titles like "Something ("
        if "(" in title and ")" not in title:
            continue

        results.append(
            WebPageResult(
                title=title,
                url=url,
                reason=f"This page was found while investigating: {query}",
            )
        )

    return results


def collect_wikipedia_pages(search_angles: List[str]) -> List[WebPageResult]:
    search_tool = DuckDuckGoSearchTool()
    seen_urls = set()
    pages: List[WebPageResult] = []

    for angle in search_angles:
        print(f"Searching: {angle}")
        for page in wikipedia_search(search_tool, angle):
            if page.url in seen_urls:
                continue
            seen_urls.add(page.url)
            pages.append(page)

    return pages


def synthesize_answer(agent: CodeAgent, question: str, pages: List[WebPageResult]) -> str:
    page_list_text = "\n".join(
        f"- {p.title}: {p.url} -- {p.reason}"
        for p in pages[:MAX_PAGES_IN_SYNTHESIS]
    )

    prompt = f"""
Use final_answer to return a concise academic synthesis.

Research question:
{question}

Relevant Wikipedia pages found:
{page_list_text}

Write:
1. A direct answer to the question.
2. A short explanation of the main vulnerability themes.
3. A markdown list of the most relevant Wikipedia pages and URLs.

Return the answer by calling final_answer("...").
"""

    return str(agent.run(prompt))


def save_markdown_report(report: ResearchReport) -> None:
    lines: List[str] = []

    lines.append("# Research Report")
    lines.append("")
    lines.append("## Question")
    lines.append(report.topic)
    lines.append("")
    lines.append("## Computed Search Angles")
    for angle in report.search_angles:
        lines.append(f"- {angle}")

    lines.append("")
    lines.append("## Synthesis")
    lines.append(report.synthesis)
    lines.append("")
    lines.append("## Wikipedia Pages")
    for page in report.wikipedia_pages:
        lines.append(f"- [{page.title}]({page.url}) — {page.reason}")

    with open(MARKDOWN_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_report(report: ResearchReport) -> None:
    with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=4, ensure_ascii=False)


def main() -> int:
    agent = load_agent(max_steps=2)

    print("Computing search angles with Qwen CodeAgent...")
    search_angles = compute_search_angles(agent, RESEARCH_QUESTION)

    print("\nCollecting Wikipedia pages...")
    pages = collect_wikipedia_pages(search_angles)

    print("\nSynthesizing answer with Qwen CodeAgent...")
    synthesis = synthesize_answer(agent, RESEARCH_QUESTION, pages)

    report = ResearchReport(
        topic=RESEARCH_QUESTION,
        search_angles=search_angles,
        wikipedia_pages=pages,
        synthesis=synthesis,
    )

    save_json_report(report)
    save_markdown_report(report)

    print(f"\nSaved JSON report to: {JSON_OUTPUT_FILE}")
    print(f"Saved markdown report to: {MARKDOWN_OUTPUT_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
