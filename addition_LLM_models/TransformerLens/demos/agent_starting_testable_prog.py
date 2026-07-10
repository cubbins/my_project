import json
import re
import ast
import sqlite3
import torch
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer
from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DB_FILE = "research_archive.db"

RESEARCH_QUESTION = "Are liberal democracies portrayed as more vulnerable to AI competition?"


class SearchResult(BaseModel):
    title: str
    url: str
    reason: str


class ResearchState(BaseModel):
    question: str
    search_angles: List[str] = []
    search_results: List[SearchResult] = []
    theory_notes: str = ""
    evidence_notes: str = ""
    critique: str = ""
    final_report: str = ""


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


def agent_text(agent: CodeAgent, prompt: str) -> str:
    return str(agent.run(prompt)).strip()


def parse_list(raw: str) -> list[str]:
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^[-*\d.)\s]+", "", line.strip())
        if line:
            lines.append(line)

    return lines


def coordinator_agent(agent: CodeAgent, state: ResearchState) -> ResearchState:
    prompt = f"""
Return only a Python list of 5 search angles.

Research question:
{state.question}

The angles should cover:
- liberal democracy
- AI competition
- authoritarian advantage
- democratic vulnerability
- disinformation
- surveillance
- AI governance
"""

    raw = agent_text(agent, prompt)
    angles = parse_list(raw)

    if len(angles) < 3:
        angles = [
            "liberal democracy AI competition",
            "artificial intelligence authoritarianism democracy",
            "algorithmic disinformation democracy",
            "AI governance democratic states",
            "digital authoritarianism artificial intelligence",
        ]

    state.search_angles = angles[:5]
    return state


def search_agent(state: ResearchState) -> ResearchState:
    search_tool = DuckDuckGoSearchTool()
    seen_urls = set()
    results = []

    for angle in state.search_angles:
        print(f"Search Agent searching: {angle}")

        raw = search_tool.forward(f"site:en.wikipedia.org/wiki {angle}")
        text = str(raw)

        urls = re.findall(
            r"https?://(?:en\.)?(?:m\.)?wikipedia\.org/wiki/[^\s\]\)\"']+",
            text,
        )

        for url in urls:
            url = url.replace("https://en.m.wikipedia.org", "https://en.wikipedia.org")
            url = url.replace("http://en.m.wikipedia.org", "https://en.wikipedia.org")
            url = url.split("#")[0]

            if url in seen_urls:
                continue

            title = url.rsplit("/", 1)[-1].replace("_", " ")

            if "(" in title and ")" not in title:
                continue

            seen_urls.add(url)

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    reason=f"Found while searching angle: {angle}",
                )
            )

    state.search_results = results
    return state


def theory_agent(agent: CodeAgent, state: ResearchState) -> ResearchState:
    prompt = f"""
Use final_answer to return a short theory note.

Research question:
{state.question}

Search angles:
{json.dumps(state.search_angles, indent=2)}

Explain which political science / international relations concepts are relevant:
- liberal democracy
- state capacity
- democratic openness
- authoritarian control
- information manipulation
- AI governance
"""

    state.theory_notes = agent_text(agent, prompt)
    return state


def evidence_agent(agent: CodeAgent, state: ResearchState) -> ResearchState:
    pages = "\n".join(
        f"- {p.title}: {p.url} -- {p.reason}"
        for p in state.search_results[:20]
    )

    prompt = f"""
Use final_answer to evaluate the evidence.

Research question:
{state.question}

Wikipedia pages found:
{pages}

Identify which pages are most useful and why.
Do not invent pages.
"""

    state.evidence_notes = agent_text(agent, prompt)
    return state


def critique_agent(agent: CodeAgent, state: ResearchState) -> ResearchState:
    prompt = f"""
Use final_answer to critique the current research.

Question:
{state.question}

Theory notes:
{state.theory_notes}

Evidence notes:
{state.evidence_notes}

Wikipedia results:
{json.dumps([p.model_dump() for p in state.search_results[:15]], indent=2)}

Evaluate:
1. What is strong?
2. What is weak?
3. What should be searched next?
4. Whether the evidence actually answers the question.
"""

    state.critique = agent_text(agent, prompt)
    return state


def report_agent(agent: CodeAgent, state: ResearchState) -> ResearchState:
    pages = "\n".join(
        f"- [{p.title}]({p.url}) — {p.reason}"
        for p in state.search_results[:20]
    )

    prompt = f"""
Use final_answer to write a concise markdown research report.

Question:
{state.question}

Theory notes:
{state.theory_notes}

Evidence notes:
{state.evidence_notes}

Critique:
{state.critique}

Wikipedia pages:
{pages}

Report sections:
# Research Question
# Short Answer
# Theoretical Interpretation
# Evidence
# Limitations
# Wikipedia Pages
"""

    state.final_report = agent_text(agent, prompt)
    return state


def init_archive() -> None:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ResearchRun (
        RunID INTEGER PRIMARY KEY AUTOINCREMENT,
        Question TEXT,
        SearchAngles TEXT,
        TheoryNotes TEXT,
        EvidenceNotes TEXT,
        Critique TEXT,
        FinalReport TEXT,
        CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS SearchResult (
        ResultID INTEGER PRIMARY KEY AUTOINCREMENT,
        RunID INTEGER,
        Title TEXT,
        URL TEXT,
        Reason TEXT,
        FOREIGN KEY (RunID) REFERENCES ResearchRun(RunID)
    )
    """)

    conn.commit()
    conn.close()


def archive_state(state: ResearchState) -> int:
    init_archive()

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO ResearchRun
        (Question, SearchAngles, TheoryNotes, EvidenceNotes, Critique, FinalReport)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            state.question,
            json.dumps(state.search_angles, ensure_ascii=False),
            state.theory_notes,
            state.evidence_notes,
            state.critique,
            state.final_report,
        ),
    )

    run_id = cur.lastrowid

    for result in state.search_results:
        cur.execute(
            """
            INSERT INTO SearchResult
            (RunID, Title, URL, Reason)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, result.title, result.url, result.reason),
        )

    conn.commit()
    conn.close()

    return run_id


def main() -> int:
    agent = load_agent(max_steps=2)

    state = ResearchState(question=RESEARCH_QUESTION)

    print("\n[1] Research Coordinator")
    state = coordinator_agent(agent, state)

    print("\n[2] Search Agent")
    state = search_agent(state)

    print("\n[3] Theory Agent")
    state = theory_agent(agent, state)

    print("\n[4] Evidence Agent")
    state = evidence_agent(agent, state)

    print("\n[5] Critique Agent")
    state = critique_agent(agent, state)

    print("\n[6] Report Agent")
    state = report_agent(agent, state)

    print("\n[7] SQL Research Archive")
    run_id = archive_state(state)

    with open("multi_agent_research_state.json", "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=4, ensure_ascii=False)

    with open("multi_agent_research_report.md", "w", encoding="utf-8") as f:
        f.write(state.final_report)

    print(f"\nSaved archive run ID: {run_id}")
    print("Saved: multi_agent_research_state.json")
    print("Saved: multi_agent_research_report.md")
    print(f"Saved SQLite archive: {DB_FILE}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())