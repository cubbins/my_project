#!/usr/bin/env python3
"""
Qwen_IR_feature_coding_sql_rare_earth.py

Phase 3+ extension for the Qwen Wikipedia International Relations pipeline,
with added support for Rare Earth Geopolitics.

New table (same as before, plus rare earth fields):

    dbo.EventIRFeatureCoding

New rare earth feature codings
------------------------------
- rare_earth_dependency_score: 0.0–1.0
- rare_earth_leverage_score: 0.0–1.0
- rare_earth_supply_chain_stage:
    mining, processing, refining, magnet_production, manufacturing, unclear
- rare_earth_policy_type:
    export_control, diversification, stockpiling, investment, alliance,
    environmental_regulation, unclear
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
from typing import Any, Optional

import torch


IR_EVENTS = [
    "Cuban Missile Crisis",
    "Berlin Blockade",
    "Suez Crisis",
    "Iran hostage crisis",
    "Korean War",
    "Vietnam War",
    "Sino-Soviet split",
    "Prague Spring",
    "Soviet invasion of Afghanistan",
    "Able Archer 83",
    "Gulf War",
    "Bosnian War",
    "Kosovo War",
    "Rwandan genocide",
    "Iraq War",
    "War in Afghanistan (2001–2021)",
    "Russo-Ukrainian War",
    "Annexation of Crimea by the Russian Federation",
    "Russian invasion of Ukraine",
    "Syrian civil war",
    "Arab Spring",
    "Israeli–Palestinian conflict",
    "Six-Day War",
    "Yom Kippur War",
    "Camp David Accords",
    "Oslo Accords",
    "Iran–Iraq War",
    "Indo-Pakistani war of 1971",
    "Kargil War",
    "South China Sea disputes",
    "First Taiwan Strait Crisis",
    "Second Taiwan Strait Crisis",
    "Third Taiwan Strait Crisis",
    "Falklands War",
    "Yugoslav Wars",
    "Treaty of Versailles",
    "Munich Agreement",
    "Yalta Conference",
    "Potsdam Conference",
    "Nuclear Non-Proliferation Treaty",
]

RARE_EARTH_EVENTS = [
    "China rare earth export restrictions",
    "China–Japan rare earth dispute",
    "China – Rare Earths WTO dispute",
    "Mountain Pass rare earth mine",
    "Lynas rare earth processing controversy",
    "Myanmar rare earth mining",
    "Greenland rare earth mining controversy",
    "EU Critical Raw Materials Act",
    "Gallium and germanium export controls",
    "Lithium Triangle",
]

IR_EVENTS = IR_EVENTS + RARE_EARTH_EVENTS

ALLOWED_POLARITY = {
    "unipolar",
    "bipolar",
    "multipolar",
    "regional",
    "unclear",
}

ALLOWED_CONFLICT_TYPE = {
    "interstate",
    "civil",
    "proxy",
    "hybrid",
    "diplomatic_crisis",
    "treaty_institutional",
    "unclear",
}

ALLOWED_ESCALATION_STAGE = {
    "latent",
    "crisis",
    "militarized_crisis",
    "war",
    "settlement",
    "postwar_order",
    "unclear",
}

ALLOWED_QUALITY_FLAGS = {
    "OK",
    "LOW_DETAIL",
    "PLACEHOLDER",
    "PARSE_RECOVERY",
    "FAILED",
}

ALLOWED_RE_STAGE = {
    "mining",
    "processing",
    "refining",
    "magnet_production",
    "manufacturing",
    "unclear",
}

ALLOWED_RE_POLICY = {
    "export_control",
    "diversification",
    "stockpiling",
    "investment",
    "alliance",
    "environmental_regulation",
    "unclear",
}


@dataclass(frozen=True)
class SqlConfig:
    server: str
    database: str
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    trust_server_certificate: bool = True
    timeout_seconds: int = 10

    @classmethod
    def from_environment(cls, database_override: Optional[str] = None) -> "SqlConfig":
        server = os.getenv("MSSQL_SERVER")
        username = os.getenv("MSSQL_USERNAME")
        password = os.getenv("MSSQL_PASSWORD")
        database = database_override or os.getenv("LLM_DATABASE", "LLMResearch")

        missing = [
            name
            for name, value in {
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
                + "  export LLM_DATABASE='LLMResearch'\n"
            )

        return cls(
            server=str(server),
            database=str(database),
            username=str(username),
            password=str(password),
        )

    def connection_string(self) -> str:
        trust_value = "yes" if self.trust_server_certificate else "no"

        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate={trust_value};"
        )


@dataclass(frozen=True)
class AppConfig:
    model_name: str
    database: str
    max_new_tokens: int
    device: str
    dtype_name: str
    dry_run: bool
    max_events: int
    delay_seconds: float
    event_file: Optional[str]
    show_latest: bool
    skip_smoke_test: bool


def parse_args(argv: Optional[list[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Retrieve Wikipedia IR and rare earth events and store quantitative feature codings in SQL Server."
    )

    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--database", default=os.getenv("LLM_DATABASE", "LLMResearch"))
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--event-file", default=None)
    parser.add_argument("--show-latest", action="store_true")
    parser.add_argument("--skip-smoke-test", action="store_true")

    args = parser.parse_args(argv)

    return AppConfig(
        model_name=args.model,
        database=args.database,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype_name=args.dtype,
        dry_run=args.dry_run,
        max_events=args.max_events,
        delay_seconds=args.delay_seconds,
        event_file=args.event_file,
        show_latest=args.show_latest,
        skip_smoke_test=args.skip_smoke_test,
    )


def load_event_titles(event_file: Optional[str]) -> list[str]:
    if event_file is None:
        return IR_EVENTS

    path = os.path.expanduser(event_file)

    with open(path, "r", encoding="utf-8") as f:
        titles = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not titles:
        raise RuntimeError(f"No event titles found in {event_file}")

    return titles


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("You requested CUDA, but torch.cuda.is_available() is False.")

    return device_arg


def resolve_dtype(dtype_arg: str, device: str) -> torch.dtype:
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "float32":
        return torch.float32

    return torch.float16 if device == "cuda" else torch.float32


def connect_to_database(sql_config: SqlConfig):
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError(
            "pyodbc is not installed. Install it with: conda install -c conda-forge pyodbc"
        ) from exc

    return pyodbc.connect(
        sql_config.connection_string(),
        timeout=sql_config.timeout_seconds,
    )


def ensure_feature_table(sql_config: SqlConfig) -> None:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()

    cur.execute(
        """
        IF OBJECT_ID('dbo.EventIRFeatureCoding', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.EventIRFeatureCoding
            (
                FeatureCodingID        INT IDENTITY(1,1) NOT NULL PRIMARY KEY,

                EventName              NVARCHAR(300) NOT NULL,
                ModelName              NVARCHAR(300) NOT NULL,
                WikipediaSummary       NVARCHAR(MAX) NULL,

                Prompt                 NVARCHAR(MAX) NULL,
                OutputText             NVARCHAR(MAX) NULL,
                ParsedJSON             NVARCHAR(MAX) NULL,

                Polarity               NVARCHAR(50) NULL,
                ConflictType           NVARCHAR(80) NULL,
                EscalationStage        NVARCHAR(80) NULL,

                CoerciveMilitary       BIT NULL,
                CoerciveEconomic       BIT NULL,
                CoerciveDiplomatic     BIT NULL,
                CoerciveInformational  BIT NULL,

                AllianceStructure      NVARCHAR(MAX) NULL,
                StrategicObjectives    NVARCHAR(MAX) NULL,
                OutcomeSuccessScore    FLOAT NULL,
                CounterfactualOptions  NVARCHAR(MAX) NULL,
                EvidenceText           NVARCHAR(MAX) NULL,

                RealismScore           INT NULL,
                LiberalismScore        INT NULL,
                ConstructivismScore    INT NULL,

                RareEarthDependencyScore FLOAT NULL,
                RareEarthLeverageScore   FLOAT NULL,
                RareEarthSupplyChainStage NVARCHAR(80) NULL,
                RareEarthPolicyType       NVARCHAR(80) NULL,

                QualityFlag            NVARCHAR(50) NULL,
                CreatedAt              DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
            );
        END;
        """
    )

    conn.commit()
    conn.close()


def verify_sql_schema(sql_config: SqlConfig) -> None:
    ensure_feature_table(sql_config)

    required_tables = {"EventIRFeatureCoding"}

    conn = connect_to_database(sql_config)
    cur = conn.cursor()

    cur.execute("SELECT @@VERSION;")
    version = cur.fetchone()[0]
    print("Connected to SQL Server.")
    print(version.splitlines()[0])

    cur.execute(
        """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_TYPE = 'BASE TABLE';
        """
    )

    existing_tables = {row.TABLE_NAME for row in cur.fetchall()}
    print("Existing dbo tables:", sorted(existing_tables))

    missing_tables = required_tables - existing_tables
    if missing_tables:
        conn.close()
        raise RuntimeError("Missing tables: " + ", ".join(sorted(missing_tables)))

    cur.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = 'EventIRFeatureCoding';
        """
    )

    existing_columns = {row.COLUMN_NAME for row in cur.fetchall()}

    expected_columns = {
        "FeatureCodingID",
        "EventName",
        "ModelName",
        "WikipediaSummary",
        "Prompt",
        "OutputText",
        "ParsedJSON",
        "Polarity",
        "ConflictType",
        "EscalationStage",
        "CoerciveMilitary",
        "CoerciveEconomic",
        "CoerciveDiplomatic",
        "CoerciveInformational",
        "AllianceStructure",
        "StrategicObjectives",
        "OutcomeSuccessScore",
        "CounterfactualOptions",
        "EvidenceText",
        "RealismScore",
        "LiberalismScore",
        "ConstructivismScore",
        "RareEarthDependencyScore",
        "RareEarthLeverageScore",
        "RareEarthSupplyChainStage",
        "RareEarthPolicyType",
        "QualityFlag",
        "CreatedAt",
    }

    missing_columns = expected_columns - existing_columns
    if missing_columns:
        conn.close()
        raise RuntimeError(
            "dbo.EventIRFeatureCoding is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    print("EventIRFeatureCoding columns:", sorted(existing_columns))
    print("Required SQL schema is present.")
    conn.close()


def load_qwen_model(model_name: str, device: str, dtype: torch.dtype):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Install it with: pip install transformers accelerate"
        ) from exc

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
    ids_to_decode = output_ids[0, prompt_len:]
    return tokenizer.decode(ids_to_decode, skip_special_tokens=True).strip()


def run_smoke_test(tokenizer, model) -> None:
    test_prompt = "Reply with exactly one sentence and no explanation: What is the capital of France?"
    output = run_llm(tokenizer, model, test_prompt, max_new_tokens=30)

    print("\nModel smoke test")
    print("-" * 80)
    print("Prompt:", test_prompt)
    print("Output:", output)
    print("-" * 80)


def fetch_wikipedia_summary(title: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is not installed. Install it with: pip install requests") from exc

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
    }
    headers = {"User-Agent": "IR-Feature-Coding-Qwen-SQL-RareEarth/1.0"}

    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()

    data = response.json()
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))

    if "missing" in page:
        raise ValueError(f'Wikipedia article "{title}" was not found.')

    return page.get("extract", "")


def build_feature_prompt(event_name: str, summary: str) -> str:
    return f"""
Analyze this International Relations event using only the source text.

Event:
{event_name}

Source:
{summary}

Focus especially on:
- polarity, conflict type, escalation stage
- coercive instruments (military, economic, diplomatic, informational)
- alliance structure and strategic objectives
- outcome success and counterfactual policy options
- realism, liberalism, constructivism fit
- rare earth geopolitics if applicable:
  - supply chain stage (mining, processing, refining, magnet_production, manufacturing)
  - dependency and leverage
  - policy type (export_control, diversification, stockpiling, investment, alliance, environmental_regulation)

Return one JSON object only.

Use this exact JSON structure:

{{
  "polarity": "bipolar",
  "conflict_type": "diplomatic_crisis",
  "escalation_stage": "crisis",
  "coercive_instruments": {{
    "military": true,
    "economic": false,
    "diplomatic": true,
    "informational": false
  }},
  "rare_earth_dependency_score": 0.0,
  "rare_earth_leverage_score": 0.0,
  "rare_earth_supply_chain_stage": "unclear",
  "rare_earth_policy_type": "unclear",
  "alliance_structure": "Short description.",
  "strategic_objectives": "Short description.",
  "outcome_success_score": 0.75,
  "counterfactual_policy_options": "Short description.",
  "evidence_text": "Short supporting evidence from the source.",
  "realism_score": 90,
  "liberalism_score": 45,
  "constructivism_score": 60,
  "quality_flag": "OK"
}}

Allowed values:
polarity = unipolar, bipolar, multipolar, regional, unclear
conflict_type = interstate, civil, proxy, hybrid, diplomatic_crisis, treaty_institutional, unclear
escalation_stage = latent, crisis, militarized_crisis, war, settlement, postwar_order, unclear
quality_flag = OK, LOW_DETAIL, PLACEHOLDER

rare_earth_supply_chain_stage = mining, processing, refining, magnet_production, manufacturing, unclear
rare_earth_policy_type = export_control, diversification, stockpiling, investment, alliance, environmental_regulation, unclear

Scores:
realism_score, liberalism_score, constructivism_score: integers 0 to 100.
outcome_success_score: number 0.0 to 1.0.
rare_earth_dependency_score: number 0.0 to 1.0.
rare_earth_leverage_score: number 0.0 to 1.0.

No markdown. No explanation. JSON only.
""".strip()


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_objects(text: Any) -> list[dict[str, Any]]:
    if isinstance(text, dict):
        return [text]

    if not isinstance(text, str):
        return []

    cleaned = strip_markdown_fences(text)
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []

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


def is_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    return text in {"", "...", "N/A", "None", "null"}


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
        "rare_earth_dependency_score": normalize_float_score(obj.get("rare_earth_dependency_score")),
        "rare_earth_leverage_score": normalize_float_score(obj.get("rare_earth_leverage_score")),
        "rare_earth_supply_chain_stage": normalize_enum(
            obj.get("rare_earth_supply_chain_stage"), ALLOWED_RE_STAGE
        ),
        "rare_earth_policy_type": normalize_enum(
            obj.get("rare_earth_policy_type"), ALLOWED_RE_POLICY
        ),
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

    text_fields = [
        "alliance_structure",
        "strategic_objectives",
        "counterfactual_policy_options",
        "evidence_text",
    ]

    if any(is_placeholder(normalized[field]) for field in text_fields):
        normalized["quality_flag"] = "PLACEHOLDER"

    missing_core = [
        field
        for field in ["realism_score", "liberalism_score", "constructivism_score", "outcome_success_score"]
        if normalized[field] is None
    ]

    if missing_core:
        normalized["quality_flag"] = "LOW_DETAIL"

    if all(
        normalized[field] is None
        for field in [
            "coercive_military",
            "coercive_economic",
            "coercive_diplomatic",
            "coercive_informational",
        ]
    ):
        normalized["quality_flag"] = "LOW_DETAIL"

    return normalized


def safe_parse_feature_json(text: str) -> Optional[dict[str, Any]]:
    try:
        objects = extract_json_objects(text)

        for obj in objects:
            normalized = normalize_feature_json(obj)
            if normalized is not None:
                return normalized

    except Exception:
        pass

    return None


def store_feature_coding(
    sql_config: SqlConfig,
    event_name: str,
    model_name: str,
    summary: str,
    prompt: str,
    output_text: str,
    parsed: dict[str, Any],
) -> int:
    parsed_json_text = json.dumps(parsed, ensure_ascii=False)

    conn = connect_to_database(sql_config)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(
            """
            INSERT INTO dbo.EventIRFeatureCoding
                (
                    EventName,
                    ModelName,
                    WikipediaSummary,
                    Prompt,
                    OutputText,
                    ParsedJSON,
                    Polarity,
                    ConflictType,
                    EscalationStage,
                    CoerciveMilitary,
                    CoerciveEconomic,
                    CoerciveDiplomatic,
                    CoerciveInformational,
                    AllianceStructure,
                    StrategicObjectives,
                    OutcomeSuccessScore,
                    CounterfactualOptions,
                    EvidenceText,
                    RealismScore,
                    LiberalismScore,
                    ConstructivismScore,
                    RareEarthDependencyScore,
                    RareEarthLeverageScore,
                    RareEarthSupplyChainStage,
                    RareEarthPolicyType,
                    QualityFlag
                )
            OUTPUT INSERTED.FeatureCodingID
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            event_name,
            model_name,
            summary,
            prompt,
            output_text,
            parsed_json_text,
            parsed.get("polarity"),
            parsed.get("conflict_type"),
            parsed.get("escalation_stage"),
            parsed.get("coercive_military"),
            parsed.get("coercive_economic"),
            parsed.get("coercive_diplomatic"),
            parsed.get("coercive_informational"),
            parsed.get("alliance_structure"),
            parsed.get("strategic_objectives"),
            parsed.get("outcome_success_score"),
            parsed.get("counterfactual_policy_options"),
            parsed.get("evidence_text"),
            parsed.get("realism_score"),
            parsed.get("liberalism_score"),
            parsed.get("constructivism_score"),
            parsed.get("rare_earth_dependency_score"),
            parsed.get("rare_earth_leverage_score"),
            parsed.get("rare_earth_supply_chain_stage"),
            parsed.get("rare_earth_policy_type"),
            parsed.get("quality_flag"),
        )

        new_id = int(cur.fetchone()[0])
        conn.commit()
        return new_id

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def print_latest_rows(sql_config: SqlConfig) -> None:
    conn = connect_to_database(sql_config)
    cur = conn.cursor()

    print("\nRecent EventIRFeatureCoding rows:")
    cur.execute(
        """
        SELECT TOP 20
            FeatureCodingID,
            EventName,
            Polarity,
            ConflictType,
            EscalationStage,
            RealismScore,
            LiberalismScore,
            ConstructivismScore,
            OutcomeSuccessScore,
            RareEarthDependencyScore,
            RareEarthLeverageScore,
            RareEarthSupplyChainStage,
            RareEarthPolicyType,
            QualityFlag,
            CreatedAt
        FROM dbo.EventIRFeatureCoding
        ORDER BY FeatureCodingID DESC;
        """
    )

    for row in cur.fetchall():
        print(row)

    conn.close()


def print_header(config: AppConfig, device: str, dtype: torch.dtype, sql_config: Optional[SqlConfig], event_count: int) -> None:
    print("=" * 80)
    print("Qwen IR + Rare Earth Feature Coding Runner")
    print("=" * 80)
    print(f"Model              : {config.model_name}")
    print(f"Device             : {device}")
    print(f"dtype              : {dtype}")
    print(f"CUDA available     : {torch.cuda.is_available()}")
    print(f"Events listed      : {event_count}")
    print(f"Max successful     : {'no limit' if config.max_events == 0 else config.max_events}")
    print(f"Dry run            : {config.dry_run}")

    if sql_config is not None:
        print(f"SQL server         : {sql_config.server}")
        print(f"SQL database       : {sql_config.database}")
        print(f"SQL username       : {sql_config.username}")
        print("SQL password       : ***")

    print("=" * 80)


def run_pipeline(config: AppConfig) -> int:
    event_titles = load_event_titles(config.event_file)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype_name, device)

    sql_config: Optional[SqlConfig] = None
    if not config.dry_run:
        sql_config = SqlConfig.from_environment(database_override=config.database)

    print_header(config, device, dtype, sql_config, len(event_titles))

    if sql_config is not None:
        verify_sql_schema(sql_config)
    else:
        print("Dry run selected: SQL connection and SQL inserts are skipped.")

    tokenizer, model = load_qwen_model(config.model_name, device, dtype)

    if not config.skip_smoke_test:
        run_smoke_test(tokenizer, model)

    attempted = 0
    successful = 0
    inserted_ids: list[int] = []

    print("\nBeginning feature-coding batch")
    print("=" * 80)

    for event_name in event_titles:
        if config.max_events > 0 and successful >= config.max_events:
            print("\nMaximum successful event limit reached. Halting batch.")
            break

        attempted += 1

        print("\n" + "=" * 80)
        print(f"Event {attempted}: {event_name}")
        print("=" * 80)

        try:
            summary = fetch_wikipedia_summary(event_name)
            if not summary.strip():
                print("  Skipped: empty Wikipedia summary.")
                continue

            successful += 1
            print(f"  Retrieved summary length: {len(summary):,} characters")
            print(f"  Successful event count  : {successful}")

            prompt = build_feature_prompt(event_name, summary)

            start = time.time()
            output = run_llm(
                tokenizer,
                model,
                prompt,
                max_new_tokens=config.max_new_tokens,
            )
            runtime = time.time() - start

            parsed = safe_parse_feature_json(output)

            print(f"  Runtime seconds         : {runtime:.2f}")
            print(f"  Parsed                  : {parsed is not None}")

            if parsed is None:
                print("  ----- RAW MODEL OUTPUT BEGIN -----")
                print(output)
                print("  ----- RAW MODEL OUTPUT END -------")
                continue

            print("  Feature coding:")
            print(json.dumps(parsed, indent=4, ensure_ascii=False))

            if not config.dry_run:
                assert sql_config is not None
                new_id = store_feature_coding(
                    sql_config=sql_config,
                    event_name=event_name,
                    model_name=config.model_name,
                    summary=summary,
                    prompt=prompt,
                    output_text=output,
                    parsed=parsed,
                )
                inserted_ids.append(new_id)
                print(f"  Inserted FeatureCodingID: {new_id}")

            time.sleep(config.delay_seconds)

        except Exception as exc:
            print(f"  Error processing event '{event_name}': {exc}")
            continue

    print("\nBatch complete.")
    print(f"Events attempted : {attempted}")
    print(f"Events successful: {successful}")

    if not config.dry_run and config.show_latest and sql_config is not None:
        print_latest_rows(sql_config)

    return successful


def main(argv: Optional[list[str]] = None) -> None:
    config = parse_args(argv)
    run_pipeline(config)


if __name__ == "__main__":
    main(sys.argv[1:])
