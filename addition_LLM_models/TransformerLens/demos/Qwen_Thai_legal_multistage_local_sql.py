#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pyodbc
import torch


"""
python Qwen_Thai_legal_multistage_local_sql.py \
  --source-database WikiThaiSearchDB \
  --output-database LLMResearch \
  --model Qwen/Qwen3-0.6B \
  --namespace 0 \
  --max-articles 25 \
  --min-score 15 \
  --output-html thai_legal_side_by_side_report.html

python Qwen_Thai_legal_multistage_local_sql.py \
  --source-database WikiThaiSearchDB \
  --output-database LLMResearch \
  --model Qwen/Qwen3-0.6B \
  --namespace 0 \
  --max-articles 5 \
  --min-score 25 \
  --output-html thai_legal_side_by_side_report.html

"""


DEFAULT_SOURCE_DATABASE = "WikiThaiSearchDB"
DEFAULT_OUTPUT_DATABASE = "LLMResearch"


LEGAL_KEYWORDS_THAI = [
    "กฎหมาย",
    "พระราชบัญญัติ",
    "พระราชกำหนด",
    "พระราชกฤษฎีกา",
    "รัฐธรรมนูญ",
    "ประมวลกฎหมาย",
    "ศาล",
    "ศาลฎีกา",
    "ศาลรัฐธรรมนูญ",
    "ศาลปกครอง",
    "คำพิพากษา",
    "คำวินิจฉัย",
    "คดี",
    "สิทธิ",
    "เสรีภาพ",
    "มาตรา",
    "บทบัญญัติ",
    "นิติ",
    "ตุลาการ",
]

STRONG_LEGAL_PATTERNS = [
    "พระราชบัญญัติ",
    "ประมวลกฎหมาย",
    "รัฐธรรมนูญแห่งราชอาณาจักรไทย",
    "คำวินิจฉัยศาลรัฐธรรมนูญ",
    "คำพิพากษาศาลฎีกา",
    "มาตรา ",
]



LEGAL_ANALYSIS_SECTIONS = [
    "Document Identification",
    "Thai Source Summary",
    "English Translation Summary",
    "Legal Classification",
    "Law or Court Decision Evidence",
    "Constitutional or Rights Relevance",
    "Research Usefulness",
]


ALLOWED_DOCUMENT_TYPES22 = {
    "statute",
    "constitutional_provision",
    "court_decision",
    "legal_concept",
    "legal_institution",
    "biography",
    "historical_event",
    "unclear",
}

ALLOWED_DOCUMENT_TYPES = [
    "court_decision",
    "constitutional_provision",
    "statute",
    "legal_concept",
    "legal_institution",
    "historical_event",
    "biography",
    "unclear",
]





def is_strong_legal_candidate(title: str, text: str) -> bool:
    combined = f"{title}\n{text}"
    return any(pattern in combined for pattern in STRONG_LEGAL_PATTERNS)



def normalize_document_type22(value: object) -> str:
    text = str(value or "").strip().lower()

    for allowed in ALLOWED_DOCUMENT_TYPES:
        if text == allowed:
            return allowed

    for allowed in ALLOWED_DOCUMENT_TYPES:
        if allowed in text:
            return allowed

    return "unclear"

def normalize_document_type(value: object) -> str:
    text = str(value or "").strip().lower()

    # Exact match
    if text in ALLOWED_DOCUMENT_TYPES:
        return text

    # Partial match in priority order
    for allowed in ALLOWED_DOCUMENT_TYPES:
        if allowed in text:
            return allowed

    return "unclear"




def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "yes", "1"}

def normalize_int_score(value: object) -> int:
    try:
        return max(0, min(100, int(float(value))))
    except Exception:
        return 0








@dataclass(frozen=True)
class SqlConfig:
    server: str
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    timeout_seconds: int = 30

    @classmethod
    def from_environment(cls) -> "SqlConfig":
        server = os.getenv("MSSQL_SERVER", "10.0.0.20,63451")
        username = os.getenv("MSSQL_USERNAME", "cubbins")
        password = os.getenv("MSSQL_PASSWORD")

        if not password:
            raise RuntimeError("Set MSSQL_PASSWORD before running this program.")

        return cls(server=server, username=username, password=password)

    def conn_str(self, database: str) -> str:
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "TrustServerCertificate=yes;"
        )


def connect(sql_config: SqlConfig, database: str):
    return pyodbc.connect(sql_config.conn_str(database), timeout=sql_config.timeout_seconds)


def ensure_output_schema(sql_config: SqlConfig, output_database: str) -> None:
    conn = connect(sql_config, output_database)
    cur = conn.cursor()

    cur.execute("""
    IF OBJECT_ID('dbo.ThaiLegalDocumentCandidate', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.ThaiLegalDocumentCandidate
        (
            CandidateID BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            ArticleCacheID BIGINT NOT NULL,
            PageID BIGINT NULL,
            ThaiTitle NVARCHAR(1000) NOT NULL,
            TitleNormalized NVARCHAR(1000) NOT NULL,
            NamespaceID INT NOT NULL,
            MatchScore FLOAT NOT NULL,
            MatchedKeywords NVARCHAR(MAX) NULL,
            ThaiText NVARCHAR(MAX) NOT NULL,
            ThaiCharCount INT NOT NULL,
            CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    cur.execute("""
    IF OBJECT_ID('dbo.ThaiLegalDocumentTranslation', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.ThaiLegalDocumentTranslation
        (
            TranslationID BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            CandidateID BIGINT NOT NULL,
            ModelName NVARCHAR(300) NOT NULL,
            ThaiTitle NVARCHAR(1000) NOT NULL,
            EnglishTitle NVARCHAR(1000) NULL,
            ThaiText NVARCHAR(MAX) NOT NULL,
            EnglishTranslation NVARCHAR(MAX) NOT NULL,
            TranslationPrompt NVARCHAR(MAX) NULL,
            TranslationQualityFlag NVARCHAR(50) NULL,
            CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    cur.execute("""
    IF OBJECT_ID('dbo.ThaiLegalDocumentAnalysis', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.ThaiLegalDocumentAnalysis
        (
            AnalysisID BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            CandidateID BIGINT NOT NULL,
            TranslationID BIGINT NULL,
            ModelName NVARCHAR(300) NOT NULL,

            ThaiTitle NVARCHAR(1000) NOT NULL,
            EnglishTitle NVARCHAR(1000) NULL,

            ParsedJSON NVARCHAR(MAX) NULL,
            DocumentType NVARCHAR(100) NULL,
            LegalTopic NVARCHAR(300) NULL,
            LawOrCaseName NVARCHAR(1000) NULL,
            CourtOrInstitution NVARCHAR(1000) NULL,
            ContainsLawText BIT NULL,
            ContainsCourtDecision BIT NULL,
            ConstitutionalRelevanceScore INT NULL,
            RightsRelevanceScore INT NULL,
            StatuteRelevanceScore INT NULL,
            EvidenceText NVARCHAR(MAX) NULL,
            AnalysisText NVARCHAR(MAX) NULL,
            CreatedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    conn.commit()
    conn.close()


def score_legal_relevance(title: str, text: str) -> tuple[float, list[str]]:
    combined = f"{title}\n{text}"
    matched = []
    score = 0.0

    for kw in LEGAL_KEYWORDS_THAI:
        count = combined.count(kw)
        if count:
            matched.append(kw)
            score += min(count, 10)

    if "ศาลรัฐธรรมนูญ" in combined:
        score += 20
    if "คำวินิจฉัย" in combined or "คำพิพากษา" in combined:
        score += 15
    if "พระราชบัญญัติ" in combined or "ประมวลกฎหมาย" in combined:
        score += 15
    if "มาตรา" in combined:
        score += 8

    return score, matched


def fetch_candidate_articles(
    sql_config: SqlConfig,
    source_database: str,
    min_score: float,
    max_articles: int,
    namespace: str,
) -> list[dict[str, Any]]:
    conn = connect(sql_config, source_database)
    cur = conn.cursor()

    namespace_clause = ""
    if namespace != "all":
        namespace_clause = f"AND NamespaceID = {int(namespace)}"

    cur.execute(f"""
    SELECT
        ArticleCacheID,
        PageID,
        Title,
        TitleNormalized,
        NamespaceID,
        CleanText,
        CleanCharCount
    FROM dbo.WikipediaArticleCache
    WHERE CleanText IS NOT NULL
      AND CleanCharCount >= 300
      {namespace_clause}
    ORDER BY ArticleCacheID;
    """)

    candidates = []

    for row in cur.fetchall():
        title = row.Title
        text = row.CleanText or ""
        score, matched = score_legal_relevance(title, text)

        #if score >= min_score:
        if score >= min_score and is_strong_legal_candidate(title, text):
            candidates.append({
                "ArticleCacheID": int(row.ArticleCacheID),
                "PageID": row.PageID,
                "ThaiTitle": title,
                "TitleNormalized": row.TitleNormalized,
                "NamespaceID": int(row.NamespaceID),
                "ThaiText": text,
                "ThaiCharCount": int(row.CleanCharCount),
                "MatchScore": score,
                "MatchedKeywords": matched,
            })

        if max_articles > 0 and len(candidates) >= max_articles:
            break

    conn.close()
    return candidates


def insert_candidate(
    sql_config: SqlConfig,
    output_database: str,
    candidate: dict[str, Any],
) -> int:
    conn = connect(sql_config, output_database)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO dbo.ThaiLegalDocumentCandidate
    (
        ArticleCacheID,
        PageID,
        ThaiTitle,
        TitleNormalized,
        NamespaceID,
        MatchScore,
        MatchedKeywords,
        ThaiText,
        ThaiCharCount
    )
    OUTPUT INSERTED.CandidateID
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """,
    candidate["ArticleCacheID"],
    candidate["PageID"],
    candidate["ThaiTitle"],
    candidate["TitleNormalized"],
    candidate["NamespaceID"],
    candidate["MatchScore"],
    json.dumps(candidate["MatchedKeywords"], ensure_ascii=False),
    candidate["ThaiText"],
    candidate["ThaiCharCount"])

    candidate_id = int(cur.fetchone()[0])
    conn.commit()
    conn.close()
    return candidate_id


def select_legal_excerpt(text: str, max_chars: int = 12000) -> str:
    paragraphs = re.split(r"\n\s*\n+", text or "")
    useful = []

    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 80:
            continue

        if any(kw in p for kw in LEGAL_KEYWORDS_THAI):
            useful.append(p)

        if sum(len(x) for x in useful) >= max_chars:
            break

    if not useful:
        return text[:max_chars]

    return "\n\n".join(useful)[:max_chars]


def build_translation_prompt(thai_title: str, thai_text: str) -> str:
    return f"""
You are a careful Thai-to-English legal translator.

Translate the following Thai Wikipedia legal source text into English.

Rules:
- Preserve legal names, court names, statute names, article numbers, and dates.
- If the text appears to quote a law or court decision, translate it as literally as possible.
- Do not invent facts.
- If a phrase is uncertain, mark it as [uncertain].
- Output English only.

Thai title:
{thai_title}

Thai source text:
{thai_text}
""".strip()

def build_legal_analysis_prompt(
    thai_title: str,
    thai_text: str,
    english_translation: str,
) -> str:
    return f"""
You are a legal-document research assistant studying Thai Wikipedia pages.

Use only the Thai source text and English translation below.

Thai title:
{thai_title}

Thai source text:
{thai_text}

English translation:
{english_translation}

Task:
Return exactly one JSON object with this structure:

{{
  "english_title": "English translation of the Thai title",
  "document_type": "unclear",
  "legal_topic": "short topic label",
  "law_or_case_name": "name of law, case, decision, or legal instrument if present",
  "court_or_institution": "court, legislature, ministry, or institution if present",
  "contains_law_text": true,
  "contains_court_decision": false,
  "constitutional_relevance_score": 0,
  "rights_relevance_score": 0,
  "statute_relevance_score": 0,
  "evidence_text": "specific evidence from the source",
  "research_usefulness": "why this page is useful or not useful for legal study",
  "quality_flag": "OK"
}}

For document_type, choose exactly one of:
- statute
- constitutional_provision
- court_decision
- legal_concept
- legal_institution
- biography
- historical_event
- unclear

For quality_flag, choose exactly one of:
- OK
- LOW_DETAIL
- UNCLEAR

Scoring:
- 0 means no evidence.
- 100 means very strong evidence.
- If the page states or quotes law text, contains_law_text must be true.
- If the page discusses a court judgment or decision on validity of laws, contains_court_decision must be true.

Rules:
- Return only valid JSON.
- Do not use markdown.
- Do not include explanations outside the JSON.
- Do not return the entire list of allowed values.
- Each field must contain one value only.
""".strip()

def build_side_by_side_report(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 32px; }",
        "th, td { border: 1px solid #ccc; padding: 10px; vertical-align: top; }",
        "th { background: #f2f2f2; }",
        ".thai { width: 50%; }",
        ".english { width: 50%; }",
        "</style></head><body>",
        "<h1>Thai Legal Wikipedia Side-by-Side Translation Report</h1>",
    ]

    for r in records:
        html.append(f"<h2>{r['thai_title']}</h2>")
        html.append("<table>")
        html.append("<tr><th class='thai'>Thai Source</th><th class='english'>English Translation</th></tr>")
        html.append("<tr>")
        html.append(f"<td>{escape_html(r['thai_excerpt'])}</td>")
        html.append(f"<td>{escape_html(r['english_translation'])}</td>")
        html.append("</tr>")
        html.append("</table>")
        html.append("<h3>Legal Analysis JSON</h3>")
        html.append("<pre>")
        html.append(escape_html(json.dumps(r["analysis_json"], indent=2, ensure_ascii=False)))
        html.append("</pre>")

    html.append("</body></html>")
    output_path.write_text("\n".join(html), encoding="utf-8")


def escape_html(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def load_qwen(model_name: str, device_arg: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if device_arg == "auto" and torch.cuda.is_available() else device_arg
    if device == "auto":
        device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    model.eval()
    torch.set_grad_enabled(False)

    return tokenizer, model


def run_llm(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()

    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {
        "document_type": "unclear",
        "quality_flag": "UNCLEAR",
        "evidence_text": "Could not parse model output as JSON.",
        "raw_output": text,
    }


def insert_translation(
    sql_config: SqlConfig,
    output_database: str,
    candidate_id: int,
    model_name: str,
    thai_title: str,
    english_title: Optional[str],
    thai_text: str,
    english_translation: str,
    prompt: str,
    quality_flag: str,
) -> int:
    conn = connect(sql_config, output_database)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO dbo.ThaiLegalDocumentTranslation
    (
        CandidateID,
        ModelName,
        ThaiTitle,
        EnglishTitle,
        ThaiText,
        EnglishTranslation,
        TranslationPrompt,
        TranslationQualityFlag
    )
    OUTPUT INSERTED.TranslationID
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """,
    candidate_id,
    model_name,
    thai_title,
    english_title,
    thai_text,
    english_translation,
    prompt,
    quality_flag)

    translation_id = int(cur.fetchone()[0])
    conn.commit()
    conn.close()
    return translation_id


def insert_analysis(
    sql_config: SqlConfig,
    output_database: str,
    candidate_id: int,
    translation_id: int,
    model_name: str,
    thai_title: str,
    analysis_json: dict[str, Any],
    analysis_text: str,
) -> int:
    conn = connect(sql_config, output_database)
    cur = conn.cursor()
    document_type = normalize_document_type(
        analysis_json.get("document_type")
    )

    contains_law_text = normalize_bool(
        analysis_json.get("contains_law_text")
    )

    contains_court_decision = normalize_bool(
        analysis_json.get("contains_court_decision")
    )

    constitutional_score = normalize_int_score(
        analysis_json.get("constitutional_relevance_score")
    )

    rights_score = normalize_int_score(
        analysis_json.get("rights_relevance_score")
    )

    statute_score = normalize_int_score(
        analysis_json.get("statute_relevance_score")
    )

    cur.execute("""
    INSERT INTO dbo.ThaiLegalDocumentAnalysis
    (
        CandidateID,
        TranslationID,
        ModelName,
        ThaiTitle,
        EnglishTitle,
        ParsedJSON,
        DocumentType,
        LegalTopic,
        LawOrCaseName,
        CourtOrInstitution,
        ContainsLawText,
        ContainsCourtDecision,
        ConstitutionalRelevanceScore,
        RightsRelevanceScore,
        StatuteRelevanceScore,
        EvidenceText,
        AnalysisText
    )
    OUTPUT INSERTED.AnalysisID
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """,
    candidate_id,
    translation_id,
    model_name,
    thai_title,
    analysis_json.get("english_title"),
    json.dumps(analysis_json, ensure_ascii=False),
    document_type,
    analysis_json.get("legal_topic"),
    analysis_json.get("law_or_case_name"),
    analysis_json.get("court_or_institution"),
    contains_law_text,
    contains_court_decision,
    constitutional_score,
    rights_score,
    statute_score,
    analysis_json.get("evidence_text"),
    analysis_text)

    analysis_id = int(cur.fetchone()[0])
    conn.commit()
    conn.close()
    return analysis_id


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source-database", default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--output-database", default=DEFAULT_OUTPUT_DATABASE)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--namespace", default="0", help="'0' for article namespace, or 'all'.")
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=10.0)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--max-translation-tokens", type=int, default=1200)
    parser.add_argument("--max-analysis-tokens", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-html", default="thai_legal_side_by_side_report.html")

    args = parser.parse_args()

    sql_config = SqlConfig.from_environment()

    if not args.dry_run:
        ensure_output_schema(sql_config, args.output_database)

    print("=" * 100)
    print("Thai Wikipedia Legal Document Study")
    print("=" * 100)
    print(f"Source database : {args.source_database}")
    print(f"Output database : {args.output_database}")
    print(f"Namespace       : {args.namespace}")
    print(f"Max articles    : {args.max_articles}")
    print(f"Min score       : {args.min_score}")
    print(f"Dry run         : {args.dry_run}")
    print("=" * 100)

    candidates = fetch_candidate_articles(
        sql_config=sql_config,
        source_database=args.source_database,
        min_score=args.min_score,
        max_articles=args.max_articles,
        namespace=args.namespace,
    )

    print(f"Candidate legal pages found: {len(candidates)}")

    tokenizer, model = load_qwen(args.model, args.device)

    report_records = []

    for index, candidate in enumerate(candidates, start=1):
        thai_title = candidate["ThaiTitle"]
        thai_excerpt = select_legal_excerpt(candidate["ThaiText"], args.max_source_chars)

        print("\n" + "=" * 100)
        print(f"{index}. {thai_title}")
        print(f"Score: {candidate['MatchScore']}")
        print(f"Keywords: {candidate['MatchedKeywords']}")
        print(f"Excerpt chars: {len(thai_excerpt)}")

        candidate_id = -1
        if not args.dry_run:
            candidate_id = insert_candidate(sql_config, args.output_database, candidate)
            print(f"CandidateID: {candidate_id}")

        translation_prompt = build_translation_prompt(thai_title, thai_excerpt)
        english_translation = run_llm(
            tokenizer,
            model,
            translation_prompt,
            max_new_tokens=args.max_translation_tokens,
        )

        analysis_prompt = build_legal_analysis_prompt(
            thai_title=thai_title,
            thai_text=thai_excerpt,
            english_translation=english_translation,
        )

        analysis_output = run_llm(
            tokenizer,
            model,
            analysis_prompt,
            max_new_tokens=args.max_analysis_tokens,
        )

        analysis_json = parse_json_object(analysis_output)
        english_title = analysis_json.get("english_title")
        quality_flag = analysis_json.get("quality_flag", "OK")

        translation_id = -1
        analysis_id = -1

        if not args.dry_run:
            translation_id = insert_translation(
                sql_config=sql_config,
                output_database=args.output_database,
                candidate_id=candidate_id,
                model_name=args.model,
                thai_title=thai_title,
                english_title=english_title,
                thai_text=thai_excerpt,
                english_translation=english_translation,
                prompt=translation_prompt,
                quality_flag=quality_flag,
            )

            analysis_id = insert_analysis(
                sql_config=sql_config,
                output_database=args.output_database,
                candidate_id=candidate_id,
                translation_id=translation_id,
                model_name=args.model,
                thai_title=thai_title,
                analysis_json=analysis_json,
                analysis_text=analysis_output,
            )

        print(f"TranslationID: {translation_id}")
        print(f"AnalysisID: {analysis_id}")
        print(json.dumps(analysis_json, indent=2, ensure_ascii=False))

        report_records.append({
            "thai_title": thai_title,
            "thai_excerpt": thai_excerpt,
            "english_translation": english_translation,
            "analysis_json": analysis_json,
        })

    build_side_by_side_report(Path(args.output_html), report_records)

    print("\n" + "=" * 100)
    print("Legal study run complete")
    print(f"HTML report: {args.output_html}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())