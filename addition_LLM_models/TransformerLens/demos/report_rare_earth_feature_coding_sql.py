#!/usr/bin/env python3
"""
report_rare_earth_feature_coding_sql.py

Standalone SQL Server report program for the Qwen rare-earth feature-coding runs.

Purpose
-------
Reads saved rows from:

    LLMResearch.dbo.EventIRFeatureCoding

and writes a complete readable report to a text file.

It can also check whether each rare-earth event has a cached full Wikipedia article in:

    WikipediaSearchDB.dbo.WikipediaArticleApiCache

This program does NOT call Qwen.
This program does NOT call Wikipedia.
This program only reads SQL Server.

Required environment variables
------------------------------
    MSSQL_SERVER
    MSSQL_USERNAME
    MSSQL_PASSWORD

Optional:
    LLM_DATABASE

Example environment
-------------------
    export MSSQL_SERVER="10.0.0.20,63451"
    export MSSQL_USERNAME="cubbins"
    export MSSQL_PASSWORD="your_password"
    export LLM_DATABASE="LLMResearch"

Example commands
----------------

Report the latest rare-earth rows:

    python report_rare_earth_feature_coding_sql.py \
        --database LLMResearch \
        --theme rare_earth \
        --latest 20 \
        --output-file rare_earth_feature_report.txt

Report the specific current run shown in your terminal output:

    python report_rare_earth_feature_coding_sql.py \
        --database LLMResearch \
        --theme rare_earth \
        --min-id 114 \
        --max-id 117 \
        --source-database WikipediaSearchDB \
        --output-file rare_earth_feature_report_114_117.txt

Include stored model output and prompt text:

    python report_rare_earth_feature_coding_sql.py \
        --database LLMResearch \
        --theme rare_earth \
        --min-id 114 \
        --max-id 117 \
        --include-model-output \
        --include-prompts \
        --output-file rare_earth_feature_report_full.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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
                + "  export LLM_DATABASE='LLMResearch'\n"
            )

        return cls(
            server=str(server),
            database=str(database),
            username=str(username),
            password=str(password),
        )

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
    database: str
    source_database: Optional[str]
    theme: str
    latest: int
    min_id: Optional[int]
    max_id: Optional[int]
    output_file: Path
    include_prompts: bool
    include_model_output: bool
    include_wikipedia_summary: bool
    include_parsed_json: bool


def parse_args(argv: Optional[list[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Report saved Qwen rare-earth feature-coding rows from SQL Server."
    )

    parser.add_argument("--database", default=os.getenv("LLM_DATABASE", "LLMResearch"))
    parser.add_argument(
        "--source-database",
        default=None,
        help="Optional Wikipedia cache database, usually WikipediaSearchDB.",
    )
    parser.add_argument("--theme", default="rare_earth")
    parser.add_argument("--latest", type=int, default=20)
    parser.add_argument("--min-id", type=int, default=None)
    parser.add_argument("--max-id", type=int, default=None)
    parser.add_argument("--output-file", default="rare_earth_feature_coding_report.txt")

    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--include-model-output", action="store_true")
    parser.add_argument("--include-wikipedia-summary", action="store_true")
    parser.add_argument(
        "--no-parsed-json",
        action="store_true",
        help="Do not print parsed JSON details.",
    )

    args = parser.parse_args(argv)

    return AppConfig(
        database=args.database,
        source_database=args.source_database,
        theme=args.theme,
        latest=args.latest,
        min_id=args.min_id,
        max_id=args.max_id,
        output_file=Path(args.output_file),
        include_prompts=args.include_prompts,
        include_model_output=args.include_model_output,
        include_wikipedia_summary=args.include_wikipedia_summary,
        include_parsed_json=not args.no_parsed_json,
    )


def connect(sql_config: SqlConfig, database: Optional[str] = None):
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError(
            "pyodbc is not installed. Install it with: conda install -c conda-forge pyodbc"
        ) from exc

    return pyodbc.connect(
        sql_config.connection_string(database=database),
        timeout=sql_config.timeout_seconds,
    )


def normalize_title(title: str) -> str:
    text = str(title or "").strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def quote_db_name(name: str) -> str:
    return "[" + name.replace("]", "]]") + "]"


def verify_feature_table(sql_config: SqlConfig) -> None:
    conn = connect(sql_config)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = 'dbo'
          AND TABLE_NAME = 'EventIRFeatureCoding';
        """
    )

    exists = int(cur.fetchone()[0]) > 0
    conn.close()

    if not exists:
        raise RuntimeError(
            f"dbo.EventIRFeatureCoding was not found in database {sql_config.database}."
        )


def fetch_feature_rows(sql_config: SqlConfig, config: AppConfig) -> list[dict[str, Any]]:
    conn = connect(sql_config)
    cur = conn.cursor()

    base_select = """
        SELECT
            FeatureCodingID,
            EventName,
            Theme,
            ModelName,
            Polarity,
            ConflictType,
            EscalationStage,

            CoerciveMilitary,
            CoerciveEconomic,
            CoerciveDiplomatic,
            CoerciveInformational,

            ResourceLeverageType,
            SupplyChainStage,
            ImportDependenceScore,
            ExportControlIntensity,
            ChinaCentralityScore,
            DiversificationPressureScore,
            DefenseRelevanceScore,
            TechnologyRelevanceScore,
            EnvironmentalConstraintScore,
            WTOTradeLawRelevance,

            AllianceStructure,
            StrategicObjectives,
            OutcomeSuccessScore,
            CounterfactualOptions,
            EvidenceText,

            RealismScore,
            LiberalismScore,
            ConstructivismScore,
            QualityFlag,
            CreatedAt,

            WikipediaSummary,
            Prompt,
            OutputText,
            ParsedJSON
        FROM dbo.EventIRFeatureCoding
        WHERE Theme = ?
    """

    params: list[Any] = [config.theme]

    if config.min_id is not None:
        base_select += " AND FeatureCodingID >= ?"
        params.append(config.min_id)

    if config.max_id is not None:
        base_select += " AND FeatureCodingID <= ?"
        params.append(config.max_id)

    if config.min_id is None and config.max_id is None:
        query = f"""
            SELECT TOP ({int(config.latest)}) *
            FROM
            (
                {base_select}
            ) AS RecentRows
            ORDER BY FeatureCodingID DESC;
        """
    else:
        query = base_select + " ORDER BY FeatureCodingID ASC;"

    cur.execute(query, params)

    columns = [column[0] for column in cur.description]
    rows: list[dict[str, Any]] = []

    for raw_row in cur.fetchall():
        row = {columns[i]: raw_row[i] for i in range(len(columns))}
        rows.append(row)

    conn.close()

    if config.min_id is None and config.max_id is None:
        rows = sorted(rows, key=lambda r: int(r["FeatureCodingID"]))

    return rows


def parse_json_safely(text: Any) -> Optional[dict[str, Any]]:
    if text is None:
        return None

    if isinstance(text, dict):
        return text

    s = str(text).strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def fetch_cache_status(
    sql_config: SqlConfig,
    source_database: str,
    titles: list[str],
) -> dict[str, dict[str, Any]]:
    if not titles:
        return {}

    conn = connect(sql_config, database=source_database)
    cur = conn.cursor()

    results: dict[str, dict[str, Any]] = {}

    for title in titles:
        norm = normalize_title(title)

        cur.execute(
            """
            SELECT TOP 1
                ArticleApiCacheID,
                RequestedTitle,
                ResolvedTitle,
                ExtractMode,
                ExtractCharCount,
                SourceURL,
                DateRetrieved
            FROM dbo.WikipediaArticleApiCache
            WHERE TitleNormalized = ?
            ORDER BY
                CASE WHEN ExtractMode = 'full' THEN 0 ELSE 1 END,
                DateRetrieved DESC;
            """,
            norm,
        )

        row = cur.fetchone()

        if row:
            results[norm] = {
                "ArticleApiCacheID": int(row.ArticleApiCacheID),
                "RequestedTitle": str(row.RequestedTitle),
                "ResolvedTitle": str(row.ResolvedTitle),
                "ExtractMode": str(row.ExtractMode),
                "ExtractCharCount": int(row.ExtractCharCount),
                "SourceURL": str(row.SourceURL),
                "DateRetrieved": str(row.DateRetrieved),
            }
        else:
            results[norm] = {
                "status": "not_cached",
            }

    conn.close()
    return results


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    def avg(field: str) -> Optional[float]:
        values = []
        for row in rows:
            value = row.get(field)
            if value is not None:
                try:
                    values.append(float(value))
                except Exception:
                    pass
        return sum(values) / len(values) if values else None

    quality_counts: dict[str, int] = {}
    conflict_counts: dict[str, int] = {}
    leverage_counts: dict[str, int] = {}

    for row in rows:
        quality = str(row.get("QualityFlag") or "NULL")
        conflict = str(row.get("ConflictType") or "NULL")
        leverage = str(row.get("ResourceLeverageType") or "NULL")

        quality_counts[quality] = quality_counts.get(quality, 0) + 1
        conflict_counts[conflict] = conflict_counts.get(conflict, 0) + 1
        leverage_counts[leverage] = leverage_counts.get(leverage, 0) + 1

    return {
        "count": len(rows),
        "first_id": min(int(r["FeatureCodingID"]) for r in rows),
        "last_id": max(int(r["FeatureCodingID"]) for r in rows),
        "quality_counts": quality_counts,
        "conflict_counts": conflict_counts,
        "resource_leverage_counts": leverage_counts,
        "avg_import_dependence": avg("ImportDependenceScore"),
        "avg_export_control_intensity": avg("ExportControlIntensity"),
        "avg_china_centrality": avg("ChinaCentralityScore"),
        "avg_diversification_pressure": avg("DiversificationPressureScore"),
        "avg_defense_relevance": avg("DefenseRelevanceScore"),
        "avg_technology_relevance": avg("TechnologyRelevanceScore"),
        "avg_environmental_constraint": avg("EnvironmentalConstraintScore"),
        "avg_realism": avg("RealismScore"),
        "avg_liberalism": avg("LiberalismScore"),
        "avg_constructivism": avg("ConstructivismScore"),
        "avg_outcome_success": avg("OutcomeSuccessScore"),
    }


def format_bool(value: Any) -> str:
    if value is None:
        return "NULL"
    return "true" if bool(value) else "false"


def add_wrapped_block(lines: list[str], title: str, text: Any, limit: Optional[int] = None) -> None:
    content = "" if text is None else str(text).strip()

    if limit is not None and len(content) > limit:
        content = content[:limit] + f"\n\n[TRUNCATED TO {limit:,} CHARACTERS IN REPORT]"

    lines.append(title)
    lines.append("-" * 100)
    lines.append(content if content else "(empty)")
    lines.append("")


def build_report(
    rows: list[dict[str, Any]],
    config: AppConfig,
    cache_status: Optional[dict[str, dict[str, Any]]] = None,
) -> str:
    summary = summarize_rows(rows)

    lines: list[str] = []

    lines.append("=" * 100)
    lines.append("RARE EARTH QWEN FEATURE-CODING SQL REPORT")
    lines.append("=" * 100)
    lines.append(f"Feature database       : {config.database}")
    lines.append(f"Theme                  : {config.theme}")
    lines.append(f"Row filter min-id      : {config.min_id}")
    lines.append(f"Row filter max-id      : {config.max_id}")
    lines.append(f"Latest limit           : {config.latest}")
    lines.append(f"Source cache database  : {config.source_database or '(not checked)'}")
    lines.append("=" * 100)
    lines.append("")

    if not rows:
        lines.append("No rows matched the requested filter.")
        return "\n".join(lines)

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(f"Rows reported          : {summary.get('count')}")
    lines.append(f"FeatureCodingID range  : {summary.get('first_id')} to {summary.get('last_id')}")
    lines.append(f"Quality counts         : {summary.get('quality_counts')}")
    lines.append(f"Conflict counts        : {summary.get('conflict_counts')}")
    lines.append(f"Resource leverage      : {summary.get('resource_leverage_counts')}")
    lines.append("")
    lines.append("Average scores:")
    for label, field in [
        ("Import dependence", "avg_import_dependence"),
        ("Export-control intensity", "avg_export_control_intensity"),
        ("China centrality", "avg_china_centrality"),
        ("Diversification pressure", "avg_diversification_pressure"),
        ("Defense relevance", "avg_defense_relevance"),
        ("Technology relevance", "avg_technology_relevance"),
        ("Environmental constraint", "avg_environmental_constraint"),
        ("Realism", "avg_realism"),
        ("Liberalism", "avg_liberalism"),
        ("Constructivism", "avg_constructivism"),
        ("Outcome success", "avg_outcome_success"),
    ]:
        value = summary.get(field)
        if value is None:
            lines.append(f"  {label:<30}: NULL")
        else:
            lines.append(f"  {label:<30}: {value:.2f}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("ROW DETAILS")
    lines.append("=" * 100)

    for row in rows:
        event_name = str(row.get("EventName") or "")
        norm = normalize_title(event_name)
        parsed = parse_json_safely(row.get("ParsedJSON"))

        lines.append("")
        lines.append("=" * 100)
        lines.append(f"FeatureCodingID {row.get('FeatureCodingID')}: {event_name}")
        lines.append("=" * 100)
        lines.append(f"CreatedAt                    : {row.get('CreatedAt')}")
        lines.append(f"ModelName                    : {row.get('ModelName')}")
        lines.append(f"Theme                        : {row.get('Theme')}")
        lines.append(f"QualityFlag                  : {row.get('QualityFlag')}")
        lines.append("")
        lines.append("Core coding:")
        lines.append(f"  Polarity                   : {row.get('Polarity')}")
        lines.append(f"  ConflictType               : {row.get('ConflictType')}")
        lines.append(f"  EscalationStage            : {row.get('EscalationStage')}")
        lines.append(f"  ResourceLeverageType       : {row.get('ResourceLeverageType')}")
        lines.append(f"  SupplyChainStage           : {row.get('SupplyChainStage')}")
        lines.append("")
        lines.append("Coercive instruments:")
        lines.append(f"  Military                   : {format_bool(row.get('CoerciveMilitary'))}")
        lines.append(f"  Economic                   : {format_bool(row.get('CoerciveEconomic'))}")
        lines.append(f"  Diplomatic                 : {format_bool(row.get('CoerciveDiplomatic'))}")
        lines.append(f"  Informational              : {format_bool(row.get('CoerciveInformational'))}")
        lines.append("")
        lines.append("Rare-earth / critical-minerals scores:")
        for label, field in [
            ("ImportDependenceScore", "ImportDependenceScore"),
            ("ExportControlIntensity", "ExportControlIntensity"),
            ("ChinaCentralityScore", "ChinaCentralityScore"),
            ("DiversificationPressureScore", "DiversificationPressureScore"),
            ("DefenseRelevanceScore", "DefenseRelevanceScore"),
            ("TechnologyRelevanceScore", "TechnologyRelevanceScore"),
            ("EnvironmentalConstraintScore", "EnvironmentalConstraintScore"),
            ("WTOTradeLawRelevance", "WTOTradeLawRelevance"),
        ]:
            value = row.get(field)
            if field == "WTOTradeLawRelevance":
                value = format_bool(value)
            lines.append(f"  {label:<35}: {value}")
        lines.append("")
        lines.append("IR theory scores:")
        lines.append(f"  RealismScore               : {row.get('RealismScore')}")
        lines.append(f"  LiberalismScore            : {row.get('LiberalismScore')}")
        lines.append(f"  ConstructivismScore        : {row.get('ConstructivismScore')}")
        lines.append(f"  OutcomeSuccessScore        : {row.get('OutcomeSuccessScore')}")
        lines.append("")
        lines.append("Narrative fields:")
        lines.append(f"  AllianceStructure          : {row.get('AllianceStructure')}")
        lines.append(f"  StrategicObjectives        : {row.get('StrategicObjectives')}")
        lines.append(f"  CounterfactualOptions      : {row.get('CounterfactualOptions')}")
        lines.append(f"  EvidenceText               : {row.get('EvidenceText')}")
        lines.append("")

        if cache_status is not None:
            status = cache_status.get(norm, {"status": "not_checked"})
            lines.append("Local Wikipedia source-cache status:")
            for key, value in status.items():
                lines.append(f"  {key:<24}: {value}")
            lines.append("")

        if config.include_parsed_json:
            if parsed is not None:
                add_wrapped_block(
                    lines,
                    "ParsedJSON",
                    json.dumps(parsed, indent=4, ensure_ascii=False),
                )
            else:
                add_wrapped_block(lines, "ParsedJSON", row.get("ParsedJSON"), limit=4000)

        if config.include_wikipedia_summary:
            add_wrapped_block(
                lines,
                "WikipediaSummary",
                row.get("WikipediaSummary"),
                limit=20000,
            )

        if config.include_model_output:
            add_wrapped_block(
                lines,
                "OutputText",
                row.get("OutputText"),
                limit=20000,
            )

        if config.include_prompts:
            add_wrapped_block(
                lines,
                "Prompt",
                row.get("Prompt"),
                limit=20000,
            )

    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    config = parse_args(argv)

    sql_config = SqlConfig.from_environment(database_override=config.database)

    print("=" * 100)
    print("Rare Earth Feature-Coding SQL Reporter")
    print("=" * 100)
    print(f"SQL server       : {sql_config.server}")
    print(f"Feature database : {sql_config.database}")
    print(f"Theme            : {config.theme}")
    print(f"Output file      : {config.output_file}")
    print("=" * 100)

    verify_feature_table(sql_config)

    rows = fetch_feature_rows(sql_config, config)
    print(f"Rows retrieved: {len(rows):,}")

    cache_status = None
    if config.source_database:
        print(f"Checking source cache database: {config.source_database}")
        cache_status = fetch_cache_status(
            sql_config=sql_config,
            source_database=config.source_database,
            titles=[str(row.get("EventName") or "") for row in rows],
        )

    report = build_report(rows, config, cache_status=cache_status)
    write_report(config.output_file, report)

    print(f"Report written: {config.output_file}")

    if rows:
        print("\nRows included:")
        for row in rows:
            print(f"  {row.get('FeatureCodingID')}: {row.get('EventName')}")

    return 0


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
