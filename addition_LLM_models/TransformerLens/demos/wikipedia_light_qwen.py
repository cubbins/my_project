import json
import re
import ast
import torch
from typing import List

from pydantic import BaseModel
from transformers import AutoTokenizer
from smolagents import CodeAgent, DuckDuckGoSearchTool, TransformersModel, tool


class WebPageResult(BaseModel):
    title: str
    url: str
    reason: str


class ResearchReport(BaseModel):
    topic: str
    wikipedia_pages: List[WebPageResult]


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "research_results.json"

RESEARCH_QUESTION = "Are liberal democracies portrayed as more vulnerable to AI competition?"

SEARCH_QUERIES = [
    "liberal democracy AI competition",
    "artificial intelligence authoritarianism democracy",
    "algorithmic disinformation democracy",
    "AI governance democratic states",
    "surveillance capitalism democracy",
    "computational propaganda democracy",
]


search_tool = DuckDuckGoSearchTool()


@tool
def wikipedia_search(query: str) -> list:
    """
    Search for real Wikipedia pages related to a query.

    Args:
        query: Search query.

    Returns:
        A list of dictionaries containing title, url, and reason.
    """
    raw = search_tool.forward(f"site:en.wikipedia.org/wiki {query}")
    text = str(raw)

    urls = re.findall(
        r"https?://(?:en\.)?(?:m\.)?wikipedia\.org/wiki/[^\s\]\)\"']+",
        text,
    )

    cleaned_urls = []

    for url in urls:
        url = url.replace("https://en.m.wikipedia.org", "https://en.wikipedia.org")
        url = url.replace("http://en.m.wikipedia.org", "https://en.wikipedia.org")
        url = url.split("#")[0]

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    results = []

    for url in cleaned_urls[:8]:
        title = url.rsplit("/", 1)[-1].replace("_", " ")

        if "(" in title and ")" not in title:
            continue

        results.append(
            {
                "title": title,
                "url": url,
                "reason": f"This page appeared in a Wikipedia search for: {query}",
            }
        )

    return results


def load_agent() -> CodeAgent:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "left"

    # Do NOT add a new token unless absolutely necessary.
    # This avoids embedding-resize issues.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = TransformersModel(
        model_id=MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return CodeAgent(
        tools=[wikipedia_search],
        model=model,
        add_base_tools=False,
        max_steps=6,
        additional_authorized_imports=["json"],
    )


def parse_agent_output(raw_response):
    text = str(raw_response).strip()

    if text.startswith("```json"):
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```"):
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback for Python dict-style output with single quotes
        return ast.literal_eval(text)


def main() -> int:
    agent = load_agent()

    prompt = f"""
You are a light academic research agent.

Use the tool wikipedia_search to gather real Wikipedia pages.

Research question:
{RESEARCH_QUESTION}

Search these exact queries:
{json.dumps(SEARCH_QUERIES, indent=2)}

Rules:
- wikipedia_search returns a list of dictionaries.
- Do not invent Wikipedia pages.
- Do not create hypothetical pages.
- Only use URLs returned by wikipedia_search.
- Remove duplicate URLs.
- Return only a JSON object.
- Do not use markdown fences.

Required structure:
{{
  "topic": "{RESEARCH_QUESTION}",
  "wikipedia_pages": [
    {{
      "title": "Article Title",
      "url": "https://en.wikipedia.org/wiki/...",
      "reason": "Explain why this page helps address the research question."
    }}
  ]
}}
"""

    print("Starting Qwen agent research loop...")
    raw_response = agent.run(prompt)

    try:
        data = parse_agent_output(raw_response)
        report = ResearchReport.model_validate(data)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=4, ensure_ascii=False)

        print(f"\nSuccess. Results saved to: {OUTPUT_FILE}")
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))

    except Exception as exc:
        print("\nAgent output could not be parsed as valid research JSON.")
        print(f"Error: {exc}")
        print("\nRaw agent response:")
        print(raw_response)

        print("\nNo file was saved because the agent output failed validation.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())