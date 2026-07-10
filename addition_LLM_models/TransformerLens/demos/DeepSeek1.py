#!/usr/bin/env python3
"""
Qwen_IR_rare_earth_multistage_narrative_local_sql.py

Multi-stage rare-earth / critical-minerals International Relations analysis.

This program uses ONLY local SQL Server cached Wikipedia text as source material:

    WikipediaSearchDB.dbo.WikipediaArticleApiCache

It does NOT call Wikipedia on the internet.

For each event, it performs:

1. Retrieve cached Wikipedia article from SQL Server.
2. Select the most relevant paragraphs.
3. Qwen pass #1: create structured JSON feature coding.
4. Qwen passes #2-N: create separate scholarly narrative sections:
      1. International Context
      2. Strategic Actors
      3. Economic Instruments
      4. Theoretical Interpretation
      5. Evidence
      6. Implications for International Relations
      7. Counterfactual and Policy Alternatives
      8. Conclusion
5. Store feature JSON in dbo.EventIRFeatureCoding.
6. Store each narrative section in dbo.EventIRNarrativeSection.
7. Store assembled final report in dbo.EventIRScholarlyNarrative.
8. Optionally write a readable text report file.

Required environment variables:

    MSSQL_SERVER
    MSSQL_USERNAME
    MSSQL_PASSWORD

Optional:

    LLM_DATABASE

Example dry run:

    python Qwen_IR_rare_earth_multistage_narrative_local_sql.py \
      --database LLMResearch \
      --source-database WikipediaSearchDB \
      --max-events 1 \
      --dry-run \
      --output-report rare_earth_multistage_test.txt

Example SQL run:

    python Qwen_IR_rare_earth_multistage_narrative_local_sql.py \
      --database LLMResearch \
      --source-database WikipediaSearchDB \
      --max-events 0 \
      --output-report rare_earth_multistage_report.txt
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch


RARE_EARTH_EVENTS = [
    "Rare earths trade dispute",
    "Rare-earth element",
    "Rare-earth industry in China",
    "Rare-earth industry in Australia",
    "Rare-earth industry in Vietnam",
    "2010 Senkaku boat collision incident",
    "Senkaku Islands dispute",
    "China–Japan relations",
    "United States–China trade war",
    "China–United States trade war",
    "Mountain Pass Rare Earth Mine",
    "MP Materials",
    "Lynas",
    "Mount Weld",
    "Kuannersuit",
    "Mining industry of Greenland",
    "Economy of Greenland",
    "Critical mineral raw materials",
    "Critical minerals",
    "Supply chain resilience",
]


NARRATIVE_SECTIONS = [
    {
        "number": 1,
        "title": "International Context",
        "goal": (
            "Explain the broader international setting of the event. "
            "Discuss whether the case reflects unipolar, bipolar, multipolar, or regional dynamics. "
            "Explain why the case matters to International Relations and international political economy."
        ),
    },
    {
        "number": 2,
        "title": "Strategic Actors",
        "goal": (
            "Identify the major states, firms, institutions, or coalitions in the source text. "
            "Explain their interests, constraints, dependencies, and strategic objectives."
        ),
    },
    {
        "number": 3,
        "title": "Economic Instruments",
        "goal": (
            "Analyze the economic tools involved, including export restrictions, trade disputes, "
            "industrial policy, supply-chain control, diversification, mining, processing, refining, "
            "investment, and technological dependence."
        ),
    },
    {
        "number": 4,
        "title": "Theoretical Interpretation",
        "goal": (
            "Interpret the case through realism, liberalism, and constructivism. "
            "Explain why each theory receives its score and which theory best fits the evidence."
        ),
    },
    {
        "number": 5,
        "title": "Evidence",
        "goal": (
            "Present the factual basis from the source text. Connect specific evidence to the structured variables, "
            "especially coercive instruments, supply-chain stage, China centrality, WTO relevance, "
            "defense relevance, and technology relevance."
        ),
    },
    {
        "number": 6,
        "title": "Implications for International Relations",
        "goal": (
            "Explain what the case teaches about contemporary IR: economic security, critical minerals, "
            "strategic interdependence, institutional dispute settlement, industrial capacity, and supply-chain power."
        ),
    },
    {
        "number": 7,
        "title": "Counterfactual and Policy Alternatives",
        "goal": (
            "Analyze plausible policy alternatives using only the source text and structured coding. "
            "Explain what different actors could have done differently and how that might have affected the outcome."
        ),
    },
    {
        "number": 8,
        "title": "Conclusion",
        "goal": (
            "Synthesize the major findings for political scientists and students. "
            "State the overall importance of the case and any uncertainty or limits in the available source evidence."
        ),
    },
]


ALLOWED_POLARITY = {"unipolar", "bipolar", "multipolar", "regional", "unclear"}
ALLOWED_CONFLICT_TYPE = {
    "interstate", "civil", "proxy", "hybrid", "diplomatic_crisis",
    "treaty_institutional", "economic_security", "resource_competition", "unclear",
}
ALLOWED_ESCALATION_STAGE = {
    "latent", "crisis", "militarized_crisis", "war", "settlement",
    "postwar_order", "economic_contest", "supply_chain_pressure", "unclear",
}
ALLOWED_RESOURCE_LEVERAGE_TYPE = {
    "export_control", "investment_screening", "supply_diversification",
    "industrial_policy", "trade_dispute", "environmental_permitting",
    "military_supply_chain", "technology_supply_chain", "unclear",
}
ALLOWED_SUPPLY_CHAIN_STAGE = {
    "mining", "separation_processing", "refining", "magnet_manufacturing",
    "end_use_defense", "end_use_clean_energy", "end_use_electronics",
    "full_supply_chain", "unclear",
}
ALLOWED_QUALITY_FLAGS = {"OK", "LOW_DETAIL", "PLACEHOLDER", "PARSE_RECOVERY", "FAILED"}


@dataclass(frozen=True)
class SqlConfig:
    server: str
    database: str
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    timeout_seconds: int = 30

    @classmethod
    def from_environment(cls, database_override: Optional[str] = None) -> "SqlConfig":
        server = os.getenv("MSSQL_SERVER")
        username = os.getenv("MSSQL_USERNAME")
        password = os.getenv("MSSQL_PASSWORD")
        database = database_override or os.getenv("LLM_DATABASE", "LLMResearch")

        missing = [
            name for name, value in {
                "MSSQL_SERVER": server,
                "MSSQL_USERNAME": username,
                "MSSQL_PASSWORD": password,
            }.items()
            if not value
        ]

        if missing:
            raise RuntimeError(
                "Missing SQL environment variables: "
                + ", ".join(missing)
                + "\n\nExample:\n"
                + "  export MSSQL_SERVER='10.0.0.20,63451'\n"
                + "  export MSSQL_USERNAME='cubbins'\n"
                + "  export MSSQL_PASSWORD='your_password'\n"
            )

        return cls(str(server), str(database), str(username), str(password))

    def connection_string(self, database: Optional[str] = None) -> str:
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={database or self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate=yes;"
        )


@dataclass(frozen=True)
class AppConfig:
    model_name: str
    database: str
    source_database: str
    event_file: Optional[str]
    max_events: int
    device: str
    dtype_name: str
    dry_run: bool
    skip_smoke_test: bool
    extract_mode: str
    selected_paragraph_count: int
    max_selected_chars: int
    max_new_tokens_feature: int
    max_new_tokens_section: int
    output_report: Optional[Path]
    include_source_in_report: bool
    skip_existing_narrative: bool


def parse_args(argv: Optional[list[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Multi-stage rare-earth feature coding and section-by-section narrative generation."
    )

    #parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    parser.add_argument("--database", default=os.getenv("LLM_DATABASE", "LLMResearch"))
    parser.add_argument("--source-database", default="WikipediaSearchDB")
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--max-events", type=int, default=5, help="0 means no limit.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--extract-mode", choices=["full", "intro", "any"], default="full")
    parser.add_argument("--selected-paragraph-count", type=int, default=18)
    parser.add_argument("--max-selected-chars", type=int, default=16000)
    parser.add_argument("--max-new-tokens-feature", type=int, default=900)
    parser.add_argument("--max-new-tokens-section", type=int, default=750)
    parser.add_argument("--output-report", default=None)
    parser.add_argument("--include-source-in-report", action="store_true")
    parser.add_argument(
        "--skip-existing-narrative",
        action="store_true",
        help="Skip an event if a full assembled narrative already exists for it.",
    )

    args = parser.parse_args(argv)

    return AppConfig(
        model_name=args.model,
        database=args.database,
        source_database=args.source_database,
        event_file=args.event_file,
        max_events=args.max_events,
        device=args.device,
        dtype_name=args.dtype,
        dry_run=args.dry_run,
        skip_smoke_test=args.skip_smoke_test,
        extract_mode=args.extract_mode,
        selected_paragraph_count=args.selected_paragraph_count,
        max_selected_chars=args.max_selected_chars,
        max_new_tokens_feature=args.max_new_tokens_feature,
        max_new_tokens_section=args.max_new_tokens_section,
        output_report=Path(args.output_report) if args.output_report else None,
        include_source_in_report=args.include_source_in_report,
        skip_existing_narrative=args.skip_existing_narrative,
    )


def connect_to_database(sql_config: SqlConfig, database: Optional[str] = None):
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError("pyodbc is not installed. Install it with: conda install -c conda-forge pyodbc") from exc

    return pyodbc.connect(
        sql_config.connection_string(database=database),
        timeout=sql_config.timeout_seconds,
    )


def normalize_title(title: str) -> str:
    text = str(title or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def load_event_titles(event_file: Optional[str]) -> list[str]:
    if event_file:
        path = Path(event_file).expanduser()
        with path.open("r", encoding="utf-8") as f:
            titles = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        if not titles:
            raise RuntimeError(f"No event titles found in {path}")
        return titles
    return list(RARE_EARTH_EVENTS)


def verify_source_cache(sql_config: SqlConfig, source_database: str) -> None:
    conn = connect_to_database(sql_config, database=source_database)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = 'WikipediaArticleApiCache';
        """
    )
    exists = int(cur.fetchone()[0]) > 0
    conn.close()
    if not exists:
        raise RuntimeError(f"{source_database}.dbo.WikipediaArticleApiCache does not exist.")


def ensure_output_schema(sql_config: SqlConfig) -> None:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()

    cur.execute(
        """
        IF OBJECT_ID('dbo.EventIRFeatureCoding', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.EventIRFeatureCoding
            (
                FeatureCodingID              INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                EventName                    NVARCHAR(300) NOT NULL,
                Theme                        NVARCHAR(80) NULL,
                ModelName                    NVARCHAR(300) NOT NULL,
                WikipediaSummary             NVARCHAR(MAX) NULL,
                Prompt                       NVARCHAR(MAX) NULL,
                OutputText                   NVARCHAR(MAX) NULL,
                ParsedJSON                   NVARCHAR(MAX) NULL,
                Polarity                     NVARCHAR(50) NULL,
                ConflictType                 NVARCHAR(80) NULL,
                EscalationStage              NVARCHAR(80) NULL,
                CoerciveMilitary             BIT NULL,
                CoerciveEconomic             BIT NULL,
                CoerciveDiplomatic           BIT NULL,
                CoerciveInformational        BIT NULL,
                ResourceLeverageType         NVARCHAR(100) NULL,
                SupplyChainStage             NVARCHAR(100) NULL,
                ImportDependenceScore        INT NULL,
                ExportControlIntensity       INT NULL,
                ChinaCentralityScore         INT NULL,
                DiversificationPressureScore INT NULL,
                DefenseRelevanceScore        INT NULL,
                TechnologyRelevanceScore     INT NULL,
                EnvironmentalConstraintScore INT NULL,
                WTOTradeLawRelevance         BIT NULL,
                AllianceStructure            NVARCHAR(MAX) NULL,
                StrategicObjectives          NVARCHAR(MAX) NULL,
                OutcomeSuccessScore          FLOAT NULL,
                CounterfactualOptions        NVARCHAR(MAX) NULL,
                EvidenceText                 NVARCHAR(MAX) NULL,
                RealismScore                 INT NULL,
                LiberalismScore              INT NULL,
                ConstructivismScore          INT NULL,
                QualityFlag                  NVARCHAR(50) NULL,
                CreatedAt                    DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            );
        END;
        """
    )

    feature_columns = {
        "Theme": "NVARCHAR(80) NULL",
        "ResourceLeverageType": "NVARCHAR(100) NULL",
        "SupplyChainStage": "NVARCHAR(100) NULL",
        "ImportDependenceScore": "INT NULL",
        "ExportControlIntensity": "INT NULL",
        "ChinaCentralityScore": "INT NULL",
        "DiversificationPressureScore": "INT NULL",
        "DefenseRelevanceScore": "INT NULL",
        "TechnologyRelevanceScore": "INT NULL",
        "EnvironmentalConstraintScore": "INT NULL",
        "WTOTradeLawRelevance": "BIT NULL",
    }

    for name, sql_type in feature_columns.items():
        cur.execute(
            f"""
            IF COL_LENGTH('dbo.EventIRFeatureCoding', '{name}') IS NULL
            BEGIN
                ALTER TABLE dbo.EventIRFeatureCoding ADD {name} {sql_type};
            END;
            """
        )

    cur.execute(
        """
        IF OBJECT_ID('dbo.EventIRScholarlyNarrative', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.EventIRScholarlyNarrative
            (
                NarrativeID              INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                FeatureCodingID          INT NULL,
                EventName                NVARCHAR(300) NOT NULL,
                Theme                    NVARCHAR(80) NULL,
                ModelName                NVARCHAR(300) NOT NULL,
                SourceDatabase           NVARCHAR(300) NULL,
                SourceArticleCacheID     BIGINT NULL,
                SourceResolvedTitle      NVARCHAR(500) NULL,
                SourceExtractMode        NVARCHAR(20) NULL,
                SourceURL                NVARCHAR(1000) NULL,
                NarrativePrompt          NVARCHAR(MAX) NULL,
                NarrativeText            NVARCHAR(MAX) NOT NULL,
                NarrativeWordCount       INT NULL,
                QualityNote              NVARCHAR(MAX) NULL,
                CreatedAt                DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            );
        END;
        """
    )

    cur.execute(
        """
        IF OBJECT_ID('dbo.EventIRNarrativeSection', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.EventIRNarrativeSection
            (
                SectionID                INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                FeatureCodingID          INT NULL,
                NarrativeID              INT NULL,
                EventName                NVARCHAR(300) NOT NULL,
                Theme                    NVARCHAR(80) NULL,
                ModelName                NVARCHAR(300) NOT NULL,
                SourceDatabase           NVARCHAR(300) NULL,
                SourceArticleCacheID     BIGINT NULL,
                SectionNumber            INT NOT NULL,
                SectionTitle             NVARCHAR(300) NOT NULL,
                SectionGoal              NVARCHAR(MAX) NULL,
                SectionPrompt            NVARCHAR(MAX) NULL,
                SectionText              NVARCHAR(MAX) NOT NULL,
                SectionWordCount         INT NULL,
                RuntimeSeconds           FLOAT NULL,
                QualityNote              NVARCHAR(MAX) NULL,
                CreatedAt                DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            );
        END;
        """
    )

    cur.execute(
        """
        IF NOT EXISTS
        (
            SELECT 1
            FROM sys.indexes
            WHERE name = 'IX_EventIRNarrativeSection_Event_Section'
              AND object_id = OBJECT_ID('dbo.EventIRNarrativeSection')
        )
        BEGIN
            CREATE INDEX IX_EventIRNarrativeSection_Event_Section
            ON dbo.EventIRNarrativeSection(EventName, SectionNumber);
        END;
        """
    )

    conn.commit()
    conn.close()


def has_existing_narrative(sql_config: SqlConfig, event_name: str, model_name: str) -> bool:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT TOP 1 NarrativeID
        FROM dbo.EventIRScholarlyNarrative
        WHERE EventName = ?
          AND ModelName = ?
        ORDER BY NarrativeID DESC;
        """,
        event_name,
        model_name,
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def load_qwen_model22(model_name: str, device: str, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is not installed. Install it with: pip install transformers accelerate") from exc

    print(f"Loading tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    torch.set_grad_enabled(False)
    print("Model loaded.")
    return tokenizer, model


def load_qwen_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # DeepSeek models load best with device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True
    )

    model.eval()
    torch.set_grad_enabled(False)
    print("Model loaded.")
    return tokenizer, model












def run_llm(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def run_smoke_test(tokenizer, model) -> None:
    prompt = "Reply with exactly one sentence and no explanation: What is the capital of France?"
    output = run_llm(tokenizer, model, prompt, max_new_tokens=30)
    print("\nModel smoke test")
    print("-" * 100)
    print("Prompt:", prompt)
    print("Output:", output)
    print("-" * 100)


def fetch_cached_article(
    sql_config: SqlConfig,
    source_database: str,
    title: str,
    extract_mode: str,
) -> dict[str, Any]:
    conn = connect_to_database(sql_config, database=source_database)
    cur = conn.cursor()
    norm = normalize_title(title)

    if extract_mode == "any":
        cur.execute(
            """
            SELECT TOP 1
                ArticleApiCacheID, RequestedTitle, ResolvedTitle, TitleNormalized,
                SourceURL, ExtractMode, ExtractText, ExtractCharCount, DateRetrieved
            FROM dbo.WikipediaArticleApiCache
            WHERE TitleNormalized = ?
            ORDER BY CASE WHEN ExtractMode = 'full' THEN 0 ELSE 1 END, DateRetrieved DESC;
            """,
            norm,
        )
    else:
        cur.execute(
            """
            SELECT TOP 1
                ArticleApiCacheID, RequestedTitle, ResolvedTitle, TitleNormalized,
                SourceURL, ExtractMode, ExtractText, ExtractCharCount, DateRetrieved
            FROM dbo.WikipediaArticleApiCache
            WHERE TitleNormalized = ?
              AND ExtractMode = ?
            ORDER BY DateRetrieved DESC;
            """,
            norm,
            extract_mode,
        )

    row = cur.fetchone()
    conn.close()

    if row is None:
        raise ValueError(
            f'Local SQL cache does not contain article "{title}" '
            f"in {source_database}.dbo.WikipediaArticleApiCache with mode={extract_mode}."
        )

    return {
        "ArticleApiCacheID": int(row.ArticleApiCacheID),
        "RequestedTitle": str(row.RequestedTitle),
        "ResolvedTitle": str(row.ResolvedTitle),
        "TitleNormalized": str(row.TitleNormalized),
        "SourceURL": str(row.SourceURL),
        "ExtractMode": str(row.ExtractMode),
        "ExtractText": str(row.ExtractText),
        "ExtractCharCount": int(row.ExtractCharCount),
        "DateRetrieved": str(row.DateRetrieved),
    }


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n+", text or "")
    paragraphs = []
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part).strip()
        if len(cleaned) < 80:
            continue
        if cleaned.lower() in {"see also", "references", "external links", "further reading", "notes"}:
            continue
        paragraphs.append(cleaned)
    return paragraphs


def build_relevance_terms(event_name: str) -> dict[str, float]:
    terms = {
        "rare earth": 4.0,
        "rare-earth": 4.0,
        "critical mineral": 4.0,
        "critical raw material": 4.0,
        "supply chain": 3.5,
        "export control": 3.5,
        "export restriction": 3.5,
        "export quota": 3.5,
        "processing": 3.0,
        "separation": 3.0,
        "refining": 3.0,
        "magnet": 3.0,
        "mining": 2.8,
        "mine": 2.3,
        "mineral": 2.3,
        "resource": 2.5,
        "dependency": 2.5,
        "dependence": 2.5,
        "diversification": 3.0,
        "defense": 2.8,
        "technology": 2.5,
        "environmental": 2.3,
        "tariff": 2.2,
        "trade": 2.0,
        "wto": 2.8,
        "world trade organization": 2.8,
        "china": 2.8,
        "japan": 2.0,
        "united states": 2.3,
        "european union": 2.0,
        "security": 2.0,
        "industrial policy": 2.5,
    }

    for token in re.findall(r"[A-Za-z][A-Za-z\-–']+", event_name.lower()):
        if len(token) >= 4:
            terms[token] = max(terms.get(token, 0.0), 1.6)

    return terms


def score_paragraph(paragraph: str, terms: dict[str, float]) -> float:
    lower = paragraph.lower()
    score = 0.0

    for term, weight in terms.items():
        count = lower.count(term.lower())
        if count:
            score += weight * min(count, 4)

    if re.search(r"\b(18|19|20)\d{2}\b", paragraph):
        score += 1.5
    if re.search(r"\b\d+(?:\.\d+)?\s*%", paragraph):
        score += 1.5

    length = len(paragraph)
    if 250 <= length <= 1800:
        score += 1.0
    elif length > 3000:
        score -= 1.0

    return score


def select_relevant_text(
    full_text: str,
    event_name: str,
    max_paragraphs: int,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    paragraphs = split_paragraphs(full_text)

    if not paragraphs:
        fallback = (full_text or "").strip()[:max_chars]
        return fallback, [{"paragraph_number": 1, "score": 0.0, "char_count": len(fallback)}]

    terms = build_relevance_terms(event_name)
    scored = [
        (idx, score_paragraph(paragraph, terms), paragraph)
        for idx, paragraph in enumerate(paragraphs, start=1)
    ]

    selected_indices = {1}
    for idx, _score, _paragraph in sorted(scored, key=lambda item: item[1], reverse=True):
        if len(selected_indices) >= max_paragraphs:
            break
        selected_indices.add(idx)

    selected = [item for item in scored if item[0] in selected_indices]

    blocks = []
    metadata = []
    used_chars = 0

    for idx, score, paragraph in selected:
        block = f"[Wikipedia paragraph {idx} | relevance_score={score:.2f}]\n{paragraph}"
        if blocks and used_chars + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        used_chars += len(block) + 2
        metadata.append({"paragraph_number": idx, "score": round(score, 3), "char_count": len(paragraph)})

    selected_text = "\n\n".join(blocks).strip()
    if not selected_text:
        selected_text = (full_text or "").strip()[:max_chars]
        metadata = [{"paragraph_number": 1, "score": 0.0, "char_count": len(selected_text)}]

    return selected_text, metadata


def build_feature_prompt(event_name: str, source_text: str) -> str:
    return f"""
You are an International Relations and critical-minerals geopolitics coding assistant.

Use only the source text below. Do not invent facts not present in the source.

Event or topic:
{event_name}

Source text:
{source_text}

Task:
Code this rare-earth / critical-minerals case into structured International Relations research variables.

Return exactly one JSON object only.

Use this exact JSON structure:

{{
  "polarity": "multipolar",
  "conflict_type": "resource_competition",
  "escalation_stage": "supply_chain_pressure",
  "coercive_instruments": {{
    "military": false,
    "economic": true,
    "diplomatic": true,
    "informational": false
  }},
  "resource_leverage_type": "export_control",
  "supply_chain_stage": "separation_processing",
  "import_dependence_score": 85,
  "export_control_intensity": 80,
  "china_centrality_score": 95,
  "diversification_pressure_score": 90,
  "defense_relevance_score": 80,
  "technology_relevance_score": 90,
  "environmental_constraint_score": 60,
  "wto_trade_law_relevance": true,
  "alliance_structure": "Specific description based on the source.",
  "strategic_objectives": "Specific description based on the source.",
  "outcome_success_score": 0.70,
  "counterfactual_policy_options": "Specific alternatives based on the source.",
  "evidence_text": "Specific supporting evidence from the source.",
  "realism_score": 85,
  "liberalism_score": 70,
  "constructivism_score": 40,
  "quality_flag": "OK"
}}

Allowed values:
polarity = unipolar, bipolar, multipolar, regional, unclear
conflict_type = interstate, civil, proxy, hybrid, diplomatic_crisis, treaty_institutional, economic_security, resource_competition, unclear
escalation_stage = latent, crisis, militarized_crisis, war, settlement, postwar_order, economic_contest, supply_chain_pressure, unclear
resource_leverage_type = export_control, investment_screening, supply_diversification, industrial_policy, trade_dispute, environmental_permitting, military_supply_chain, technology_supply_chain, unclear
supply_chain_stage = mining, separation_processing, refining, magnet_manufacturing, end_use_defense, end_use_clean_energy, end_use_electronics, full_supply_chain, unclear
quality_flag = OK, LOW_DETAIL, PLACEHOLDER

Rules:
- All rare-earth relevance scores must be integers from 0 to 100.
- outcome_success_score must be a number from 0.0 to 1.0.
- realism_score, liberalism_score, constructivism_score must be integers from 0 to 100.
- Avoid placeholders.
- If evidence is insufficient, use "unclear" or lower scores rather than inventing facts.

No markdown. No explanation. JSON only.
""".strip()


def build_section_prompt(
    event_name: str,
    source_text: str,
    feature_json: dict[str, Any],
    section: dict[str, Any],
    previous_sections: list[dict[str, Any]],
) -> str:
    prior = ""
    if previous_sections:
        prior_blocks = []
        for item in previous_sections:
            prior_blocks.append(f"{item['section_number']}. {item['section_title']}\n{item['section_text']}")
        prior = "\n\nPrevious sections already written:\n" + "\n\n".join(prior_blocks)

    return f"""
You are writing one section of a scholarly International Relations report for
political scientists, international political economists, and advanced students.

Use only the source text and structured feature coding below.
Do not invent facts beyond the source text.
Do not cite outside sources.
Write in paragraphs, not bullet lists.
Do not use markdown tables.

Event or topic:
{event_name}

Section to write:
{section['number']}. {section['title']}

Section goal:
{section['goal']}

Structured feature coding:
{json.dumps(feature_json, indent=2, ensure_ascii=False)}

Source text:
{source_text}
{prior}

Task:
Write only this section: "{section['title']}".

Length target:
300 to 600 words for this section.

Required:
- Translate the structured coding into political-science reasoning.
- Explain the logic and degree of relevant scores.
- Use factual evidence from the source text.
- State uncertainty if the source text is thin.
- Keep the style suitable for political scientists and students.

Output format:
Start with the exact heading:
{section['number']}. {section['title']}

Then write the section text.
""".strip()


def extract_json_objects(text: Any) -> list[dict[str, Any]]:
    if isinstance(text, dict):
        return [text]
    if not isinstance(text, str):
        return []

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    objects = []

    for start in [m.start() for m in re.finditer(r"\{", cleaned)]:
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue

    return objects


def normalize_enum(value: Any, allowed: set[str], default: str = "unclear") -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if text in allowed:
        return text
    for allowed_value in allowed:
        if allowed_value in text:
            return allowed_value
    return default


def normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "y"}:
        return True
    if text in {"false", "no", "0", "n"}:
        return False
    return None


def normalize_score(value: Any, low: int = 0, high: int = 100) -> Optional[int]:
    try:
        score = int(round(float(value)))
    except Exception:
        return None
    return max(low, min(high, score))


def normalize_float_score(value: Any, low: float = 0.0, high: float = 1.0) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    return max(low, min(high, score))


def normalize_quality_flag(value: Any) -> str:
    text = str(value or "OK").strip().upper().replace(" ", "_")
    if text in ALLOWED_QUALITY_FLAGS:
        return text
    if "PLACE" in text:
        return "PLACEHOLDER"
    if "LOW" in text:
        return "LOW_DETAIL"
    if "FAIL" in text:
        return "FAILED"
    return "OK"


def normalize_feature_json(obj: Any) -> Optional[dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    coercion = obj.get("coercive_instruments")
    if not isinstance(coercion, dict):
        coercion = {}

    normalized = {
        "polarity": normalize_enum(obj.get("polarity"), ALLOWED_POLARITY),
        "conflict_type": normalize_enum(obj.get("conflict_type"), ALLOWED_CONFLICT_TYPE),
        "escalation_stage": normalize_enum(obj.get("escalation_stage"), ALLOWED_ESCALATION_STAGE),
        "coercive_military": normalize_bool(coercion.get("military")),
        "coercive_economic": normalize_bool(coercion.get("economic")),
        "coercive_diplomatic": normalize_bool(coercion.get("diplomatic")),
        "coercive_informational": normalize_bool(coercion.get("informational")),
        "resource_leverage_type": normalize_enum(obj.get("resource_leverage_type"), ALLOWED_RESOURCE_LEVERAGE_TYPE),
        "supply_chain_stage": normalize_enum(obj.get("supply_chain_stage"), ALLOWED_SUPPLY_CHAIN_STAGE),
        "import_dependence_score": normalize_score(obj.get("import_dependence_score")),
        "export_control_intensity": normalize_score(obj.get("export_control_intensity")),
        "china_centrality_score": normalize_score(obj.get("china_centrality_score")),
        "diversification_pressure_score": normalize_score(obj.get("diversification_pressure_score")),
        "defense_relevance_score": normalize_score(obj.get("defense_relevance_score")),
        "technology_relevance_score": normalize_score(obj.get("technology_relevance_score")),
        "environmental_constraint_score": normalize_score(obj.get("environmental_constraint_score")),
        "wto_trade_law_relevance": normalize_bool(obj.get("wto_trade_law_relevance")),
        "alliance_structure": str(obj.get("alliance_structure", "")).strip(),
        "strategic_objectives": str(obj.get("strategic_objectives", "")).strip(),
        "outcome_success_score": normalize_float_score(obj.get("outcome_success_score")),
        "counterfactual_policy_options": str(obj.get("counterfactual_policy_options", "")).strip(),
        "evidence_text": str(obj.get("evidence_text", "")).strip(),
        "realism_score": normalize_score(obj.get("realism_score")),
        "liberalism_score": normalize_score(obj.get("liberalism_score")),
        "constructivism_score": normalize_score(obj.get("constructivism_score")),
        "quality_flag": normalize_quality_flag(obj.get("quality_flag")),
    }

    weak_text = {
        "", "...", "short description", "short description.",
        "short supporting evidence from the source",
        "short supporting evidence from the source.",
    }

    for field in ["alliance_structure", "strategic_objectives", "counterfactual_policy_options", "evidence_text"]:
        if normalized[field].strip().lower() in weak_text:
            normalized["quality_flag"] = "PLACEHOLDER"

    return normalized


def parse_feature_output(output: str) -> Optional[dict[str, Any]]:
    for obj in extract_json_objects(output):
        normalized = normalize_feature_json(obj)
        if normalized is not None:
            return normalized
    return None


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def store_feature_coding(
    sql_config: SqlConfig,
    event_name: str,
    theme: str,
    model_name: str,
    summary: str,
    prompt: str,
    output_text: str,
    parsed: dict[str, Any],
) -> int:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()
    conn.autocommit = False

    try:
        cur.execute(
            """
            INSERT INTO dbo.EventIRFeatureCoding
                (
                    EventName, Theme, ModelName, WikipediaSummary,
                    Prompt, OutputText, ParsedJSON,
                    Polarity, ConflictType, EscalationStage,
                    CoerciveMilitary, CoerciveEconomic, CoerciveDiplomatic, CoerciveInformational,
                    ResourceLeverageType, SupplyChainStage,
                    ImportDependenceScore, ExportControlIntensity, ChinaCentralityScore,
                    DiversificationPressureScore, DefenseRelevanceScore, TechnologyRelevanceScore,
                    EnvironmentalConstraintScore, WTOTradeLawRelevance,
                    AllianceStructure, StrategicObjectives, OutcomeSuccessScore,
                    CounterfactualOptions, EvidenceText,
                    RealismScore, LiberalismScore, ConstructivismScore, QualityFlag
                )
            OUTPUT INSERTED.FeatureCodingID
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            event_name,
            theme,
            model_name,
            summary,
            prompt,
            output_text,
            json.dumps(parsed, ensure_ascii=False),
            parsed.get("polarity"),
            parsed.get("conflict_type"),
            parsed.get("escalation_stage"),
            parsed.get("coercive_military"),
            parsed.get("coercive_economic"),
            parsed.get("coercive_diplomatic"),
            parsed.get("coercive_informational"),
            parsed.get("resource_leverage_type"),
            parsed.get("supply_chain_stage"),
            parsed.get("import_dependence_score"),
            parsed.get("export_control_intensity"),
            parsed.get("china_centrality_score"),
            parsed.get("diversification_pressure_score"),
            parsed.get("defense_relevance_score"),
            parsed.get("technology_relevance_score"),
            parsed.get("environmental_constraint_score"),
            parsed.get("wto_trade_law_relevance"),
            parsed.get("alliance_structure"),
            parsed.get("strategic_objectives"),
            parsed.get("outcome_success_score"),
            parsed.get("counterfactual_policy_options"),
            parsed.get("evidence_text"),
            parsed.get("realism_score"),
            parsed.get("liberalism_score"),
            parsed.get("constructivism_score"),
            parsed.get("quality_flag"),
        )
        feature_id = int(cur.fetchone()[0])
        conn.commit()
        return feature_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def store_assembled_narrative(
    sql_config: SqlConfig,
    feature_id: Optional[int],
    event_name: str,
    theme: str,
    model_name: str,
    source_database: str,
    source_article: dict[str, Any],
    narrative_text: str,
    quality_note: str,
) -> int:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()
    conn.autocommit = False

    try:
        cur.execute(
            """
            INSERT INTO dbo.EventIRScholarlyNarrative
                (
                    FeatureCodingID, EventName, Theme, ModelName,
                    SourceDatabase, SourceArticleCacheID, SourceResolvedTitle,
                    SourceExtractMode, SourceURL,
                    NarrativePrompt, NarrativeText, NarrativeWordCount,
                    QualityNote
                )
            OUTPUT INSERTED.NarrativeID
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            feature_id,
            event_name,
            theme,
            model_name,
            source_database,
            source_article.get("ArticleApiCacheID"),
            source_article.get("ResolvedTitle"),
            source_article.get("ExtractMode"),
            source_article.get("SourceURL"),
            "assembled from dbo.EventIRNarrativeSection rows",
            narrative_text,
            word_count(narrative_text),
            quality_note,
        )
        narrative_id = int(cur.fetchone()[0])
        conn.commit()
        return narrative_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def store_narrative_section(
    sql_config: SqlConfig,
    feature_id: Optional[int],
    narrative_id: Optional[int],
    event_name: str,
    theme: str,
    model_name: str,
    source_database: str,
    source_article: dict[str, Any],
    section: dict[str, Any],
    prompt: str,
    section_text: str,
    runtime: float,
    quality_note: str,
) -> int:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()
    conn.autocommit = False

    try:
        cur.execute(
            """
            INSERT INTO dbo.EventIRNarrativeSection
                (
                    FeatureCodingID, NarrativeID, EventName, Theme, ModelName,
                    SourceDatabase, SourceArticleCacheID,
                    SectionNumber, SectionTitle, SectionGoal,
                    SectionPrompt, SectionText, SectionWordCount,
                    RuntimeSeconds, QualityNote
                )
            OUTPUT INSERTED.SectionID
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            feature_id,
            narrative_id,
            event_name,
            theme,
            model_name,
            source_database,
            source_article.get("ArticleApiCacheID"),
            int(section["number"]),
            str(section["title"]),
            str(section["goal"]),
            prompt,
            section_text,
            word_count(section_text),
            runtime,
            quality_note,
        )
        section_id = int(cur.fetchone()[0])
        conn.commit()
        return section_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_section_narrative_id(sql_config: SqlConfig, section_ids: list[int], narrative_id: int) -> None:
    if not section_ids:
        return

    conn = connect_to_database(sql_config)
    cur = conn.cursor()
    conn.autocommit = False

    try:
        placeholders = ",".join("?" for _ in section_ids)
        cur.execute(
            f"""
            UPDATE dbo.EventIRNarrativeSection
            SET NarrativeID = ?
            WHERE SectionID IN ({placeholders});
            """,
            narrative_id,
            *section_ids,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def assemble_narrative(section_records: list[dict[str, Any]]) -> str:
    ordered = sorted(section_records, key=lambda r: int(r["section_number"]))
    return "\n\n".join(record["section_text"].strip() for record in ordered).strip()


def append_report(
    path: Path,
    event_name: str,
    source_article: dict[str, Any],
    feature_id: Optional[int],
    narrative_id: Optional[int],
    section_records: list[dict[str, Any]],
    selected_text: str,
    selection_metadata: list[dict[str, Any]],
    parsed: dict[str, Any],
    include_source: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write(f"EVENT: {event_name}\n")
        f.write("=" * 120 + "\n")
        f.write(f"FeatureCodingID       : {feature_id}\n")
        f.write(f"NarrativeID           : {narrative_id}\n")
        f.write(f"Source cache ID       : {source_article.get('ArticleApiCacheID')}\n")
        f.write(f"Source resolved title : {source_article.get('ResolvedTitle')}\n")
        f.write(f"Source mode           : {source_article.get('ExtractMode')}\n")
        f.write(f"Source chars          : {source_article.get('ExtractCharCount')}\n")
        f.write(f"Source URL            : {source_article.get('SourceURL')}\n")
        f.write(f"Selected paragraphs   : {json.dumps(selection_metadata, ensure_ascii=False)}\n")
        f.write("\nSTRUCTURED JSON\n")
        f.write("-" * 120 + "\n")
        f.write(json.dumps(parsed, indent=4, ensure_ascii=False))
        f.write("\n\nMULTI-STAGE SCHOLARLY NARRATIVE\n")
        f.write("-" * 120 + "\n")
        f.write(assemble_narrative(section_records))
        f.write("\n\nSECTION METADATA\n")
        f.write("-" * 120 + "\n")
        for record in sorted(section_records, key=lambda r: int(r["section_number"])):
            f.write(
                f"Section {record['section_number']}: {record['section_title']} | "
                f"words={record['word_count']} | runtime={record['runtime_seconds']:.2f}s | "
                f"SectionID={record.get('section_id')}\n"
            )
        f.write("\n")

        if include_source:
            f.write("SELECTED SOURCE TEXT SENT TO QWEN\n")
            f.write("-" * 120 + "\n")
            f.write(selected_text.strip())
            f.write("\n\n")


def print_header(config: AppConfig, device: str, dtype: torch.dtype) -> None:
    print("=" * 100)
    print("Qwen Rare Earth Multi-Stage Narrative Runner")
    print("=" * 100)
    print(f"Feature DB               : {config.database}")
    print(f"Source cache DB          : {config.source_database}")
    print(f"Model                    : {config.model_name}")
    print(f"Device                   : {device}")
    print(f"dtype                    : {dtype}")
    print(f"CUDA available           : {torch.cuda.is_available()}")
    print(f"Max events               : {'no limit' if config.max_events == 0 else config.max_events}")
    print(f"Dry run                  : {config.dry_run}")
    print(f"Extract mode             : {config.extract_mode}")
    print(f"Selected paragraphs      : {config.selected_paragraph_count}")
    print(f"Max selected chars       : {config.max_selected_chars}")
    print(f"Feature max tokens       : {config.max_new_tokens_feature}")
    print(f"Section max tokens       : {config.max_new_tokens_section}")
    print(f"Output report            : {config.output_report}")
    print("=" * 100)


def run_pipeline(config: AppConfig) -> int:
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype_name, device)

    print_header(config, device, dtype)

    sql_config = SqlConfig.from_environment(database_override=config.database)
    verify_source_cache(sql_config, config.source_database)

    if not config.dry_run:
        ensure_output_schema(sql_config)
    else:
        print("Dry run selected: SQL output tables are not modified.")

    if config.output_report:
        config.output_report.write_text(
            "RARE EARTH MULTI-STAGE FEATURE + NARRATIVE REPORT\n"
            + "=" * 120
            + "\n"
            + f"CreatedAt: {datetime.now().isoformat(timespec='seconds')}\n\n",
            encoding="utf-8",
        )

    tokenizer, model = load_qwen_model(config.model_name, device, dtype)
    if not config.skip_smoke_test:
        run_smoke_test(tokenizer, model)

    events = load_event_titles(config.event_file)
    attempted = 0
    retrieved = 0
    feature_parsed = 0
    inserted_feature_ids: list[int] = []
    inserted_narrative_ids: list[int] = []
    inserted_section_ids: list[int] = []
    failed: list[str] = []
    skipped: list[str] = []

    print("\nEvents selected:")
    for i, event in enumerate(events, start=1):
        marker = ""
        if config.max_events > 0 and i > config.max_events:
            marker = " [beyond max-events limit]"
        print(f"  {i:02d}. {event}{marker}")

    print("\nBeginning multi-stage batch")
    print("=" * 100)

    for event_name in events:
        attempted += 1
        if config.max_events > 0 and attempted > config.max_events:
            print("\nMaximum attempted event limit reached.")
            break

        print("\n" + "=" * 100)
        print(f"Event {attempted}: {event_name}")
        print("=" * 100)

        try:
            if config.skip_existing_narrative and not config.dry_run:
                if has_existing_narrative(sql_config, event_name, config.model_name):
                    print("  Skipped: existing assembled narrative found.")
                    skipped.append(event_name)
                    continue

            source_article = fetch_cached_article(
                sql_config=sql_config,
                source_database=config.source_database,
                title=event_name,
                extract_mode=config.extract_mode,
            )
            retrieved += 1

            print(
                f"  SQL cache source : ID={source_article['ArticleApiCacheID']}, "
                f"title={source_article['ResolvedTitle']!r}, "
                f"mode={source_article['ExtractMode']}, "
                f"chars={source_article['ExtractCharCount']:,}"
            )

            selected_text, selection_metadata = select_relevant_text(
                full_text=source_article["ExtractText"],
                event_name=event_name,
                max_paragraphs=config.selected_paragraph_count,
                max_chars=config.max_selected_chars,
            )
            print(f"  Selected chars   : {len(selected_text):,}")
            print(f"  Paragraphs       : {len(selection_metadata)}")

            feature_prompt = build_feature_prompt(event_name, selected_text)
            start = time.time()
            feature_output = run_llm(
                tokenizer,
                model,
                feature_prompt,
                max_new_tokens=config.max_new_tokens_feature,
            )
            feature_runtime = time.time() - start
            parsed = parse_feature_output(feature_output)

            print(f"  Feature runtime  : {feature_runtime:.2f}s")
            print(f"  Feature parsed   : {parsed is not None}")

            if parsed is None:
                raise RuntimeError("Qwen feature output could not be parsed as JSON.")

            feature_parsed += 1
            print("  Feature JSON:")
            print(json.dumps(parsed, indent=4, ensure_ascii=False))

            feature_id: Optional[int] = None
            narrative_id: Optional[int] = None
            section_records: list[dict[str, Any]] = []
            event_section_ids: list[int] = []

            if not config.dry_run:
                feature_id = store_feature_coding(
                    sql_config=sql_config,
                    event_name=event_name,
                    theme="rare_earth",
                    model_name=config.model_name,
                    summary=selected_text,
                    prompt=feature_prompt,
                    output_text=feature_output,
                    parsed=parsed,
                )
                inserted_feature_ids.append(feature_id)
                print(f"  FeatureCodingID  : {feature_id}")

            previous_sections: list[dict[str, Any]] = []

            for section in NARRATIVE_SECTIONS:
                print(f"  Generating section {section['number']}: {section['title']}")

                section_prompt = build_section_prompt(
                    event_name=event_name,
                    source_text=selected_text,
                    feature_json=parsed,
                    section=section,
                    previous_sections=previous_sections,
                )

                start = time.time()
                section_text = run_llm(
                    tokenizer,
                    model,
                    section_prompt,
                    max_new_tokens=config.max_new_tokens_section,
                )
                runtime = time.time() - start
                wc = word_count(section_text)

                quality_note = (
                    f"section_words={wc}; feature_quality_flag={parsed.get('quality_flag')}; "
                    f"selected_chars={len(selected_text)}; selected_paragraphs={len(selection_metadata)}"
                )

                section_id: Optional[int] = None
                if not config.dry_run:
                    section_id = store_narrative_section(
                        sql_config=sql_config,
                        feature_id=feature_id,
                        narrative_id=None,
                        event_name=event_name,
                        theme="rare_earth",
                        model_name=config.model_name,
                        source_database=config.source_database,
                        source_article=source_article,
                        section=section,
                        prompt=section_prompt,
                        section_text=section_text,
                        runtime=runtime,
                        quality_note=quality_note,
                    )
                    event_section_ids.append(section_id)
                    inserted_section_ids.append(section_id)

                record = {
                    "section_id": section_id,
                    "section_number": section["number"],
                    "section_title": section["title"],
                    "section_text": section_text,
                    "word_count": wc,
                    "runtime_seconds": runtime,
                }
                section_records.append(record)
                previous_sections.append(record)

                print(
                    f"    words={wc:,} runtime={runtime:.2f}s "
                    f"SectionID={section_id}"
                )

            assembled = assemble_narrative(section_records)
            quality_note = (
                f"assembled_words={word_count(assembled)}; "
                f"sections={len(section_records)}; "
                f"feature_quality_flag={parsed.get('quality_flag')}"
            )

            if not config.dry_run:
                narrative_id = store_assembled_narrative(
                    sql_config=sql_config,
                    feature_id=feature_id,
                    event_name=event_name,
                    theme="rare_earth",
                    model_name=config.model_name,
                    source_database=config.source_database,
                    source_article=source_article,
                    narrative_text=assembled,
                    quality_note=quality_note,
                )
                inserted_narrative_ids.append(narrative_id)
                update_section_narrative_id(sql_config, event_section_ids, narrative_id)
                print(f"  NarrativeID      : {narrative_id}")

            print(f"  Assembled words  : {word_count(assembled):,}")

            if config.output_report:
                append_report(
                    path=config.output_report,
                    event_name=event_name,
                    source_article=source_article,
                    feature_id=feature_id,
                    narrative_id=narrative_id,
                    section_records=section_records,
                    selected_text=selected_text,
                    selection_metadata=selection_metadata,
                    parsed=parsed,
                    include_source=config.include_source_in_report,
                )
                print(f"  Report appended  : {config.output_report}")

        except Exception as exc:
            failed.append(event_name)
            print(f"  ERROR processing {event_name}")
            print(f"  {type(exc).__name__}: {exc}")

    print("\n" + "=" * 100)
    print("Multi-stage batch complete")
    print("=" * 100)
    print(f"Attempted events          : {attempted if config.max_events == 0 else min(attempted, config.max_events)}")
    print(f"Retrieved source articles : {retrieved}")
    print(f"Parsed feature codings    : {feature_parsed}")
    print(f"Inserted FeatureCodingIDs : {inserted_feature_ids}")
    print(f"Inserted NarrativeIDs     : {inserted_narrative_ids}")
    print(f"Inserted SectionIDs       : {inserted_section_ids}")
    print(f"Skipped titles            : {skipped}")
    print(f"Failed titles             : {failed}")
    if config.output_report:
        print(f"Output report             : {config.output_report}")
    print("=" * 100)

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    config = parse_args(argv)
    return run_pipeline(config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print("\nERROR:", type(exc).__name__, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
