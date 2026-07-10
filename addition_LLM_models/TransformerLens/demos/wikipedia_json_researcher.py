import json
import re
from typing import List
from pydantic import BaseModel
from smolagents import DuckDuckGoSearchTool


class WebPageResult(BaseModel):
    title: str
    url: str
    reason: str


class ResearchReport(BaseModel):
    topic: str
    wikipedia_pages: List[WebPageResult]


research_question = "Are liberal democracies portrayed as more vulnerable to AI competition?"
output_file = "research_results.json"

search_queries = [
    "liberal democracy AI competition",
    "artificial intelligence authoritarianism democracy",
    "algorithmic disinformation democracy",
    "AI governance democratic states",
    "surveillance capitalism democracy",
    "computational propaganda democracy",
]

search_tool = DuckDuckGoSearchTool()


def wikipedia_search(query: str, max_results: int = 8) -> list[dict]:
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

        if "/wiki/" not in url:
            continue

        if url not in cleaned_urls:
            cleaned_urls.append(url)

    results = []

    for url in cleaned_urls[:max_results]:
        title = url.rsplit("/", 1)[-1]
        title = title.replace("_", " ")

        # Skip malformed results
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


def main() -> int:
    seen_urls = set()
    wikipedia_pages = []

    for query in search_queries:
        print(f"Searching: {query}")

        pages = wikipedia_search(query)

        for page in pages:
            url = page["url"]

            if url in seen_urls:
                continue

            seen_urls.add(url)
            wikipedia_pages.append(page)

    report = ResearchReport(
        topic=research_question,
        wikipedia_pages=wikipedia_pages,
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=4, ensure_ascii=False)

    print(f"\nSuccess. Saved {len(wikipedia_pages)} pages to: {output_file}")
    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())