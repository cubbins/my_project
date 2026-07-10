#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import pyodbc
from transformers import pipeline


warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)


DEFAULT_SOURCE_DATABASE = "WikiThaiSearchDB"
DEFAULT_TABLE = "dbo.WikipediaArticleCache"
DEFAULT_OUTPUT_TABLE = "dbo.WikipediaArticleEnglishTranslation"


"""
python translate_wikithai_article_cache.py \
  --database WikiThaiSearchDB \
  --limit 50 \
  --namespace 0 \
  --max-cleantext-chars 2500 \
  --chunk-chars 700 \
  --model Helsinki-NLP/opus-mt-th-en \
  --device -1 \
  --resume


USE WikiThaiSearchDB;
GO

SELECT TOP (50)
    TranslationID,
    ArticleCacheID,
    PageID,
    ThaiTitle,
    EnglishTitleNormalized,
    LEFT(ThaiCleanTextExcerpt, 500) AS ThaiPreview,
    LEFT(EnglishCleanTextTranslation, 500) AS EnglishPreview,
    SourceCleanCharCount,
    TranslatedThaiCharCount,
    EnglishCharCount,
    TranslationStatus,
    CreatedAt
FROM dbo.WikipediaArticleEnglishTranslation
ORDER BY TranslationID DESC;

"""





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
    return pyodbc.connect(
        sql_config.conn_str(database),
        timeout=sql_config.timeout_seconds,
    )


def ensure_translation_table(sql_config: SqlConfig, database: str) -> None:
    conn = connect(sql_config, database)
    cur = conn.cursor()

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaArticleEnglishTranslation', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.WikipediaArticleEnglishTranslation
        (
            TranslationID BIGINT IDENTITY(1,1) NOT NULL
                CONSTRAINT PK_WikipediaArticleEnglishTranslation PRIMARY KEY,

            ArticleCacheID BIGINT NOT NULL,
            PageID BIGINT NULL,

            ThaiTitle NVARCHAR(1000) NULL,
            ThaiTitleNormalized NVARCHAR(1000) NULL,
            EnglishTitleNormalized NVARCHAR(2000) NULL,

            ThaiCleanTextExcerpt NVARCHAR(MAX) NULL,
            EnglishCleanTextTranslation NVARCHAR(MAX) NULL,

            SourceCleanCharCount INT NULL,
            TranslatedThaiCharCount INT NULL,
            EnglishCharCount INT NULL,

            TranslatorModel NVARCHAR(300) NOT NULL,
            TranslationStatus NVARCHAR(50) NOT NULL,
            ErrorMessage NVARCHAR(MAX) NULL,

            CreatedAt DATETIME2 NOT NULL
                CONSTRAINT DF_WikipediaArticleEnglishTranslation_CreatedAt
                DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1
        FROM sys.indexes
        WHERE name = 'UX_WikipediaArticleEnglishTranslation_ArticleCacheID'
          AND object_id = OBJECT_ID('dbo.WikipediaArticleEnglishTranslation')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_WikipediaArticleEnglishTranslation_ArticleCacheID
        ON dbo.WikipediaArticleEnglishTranslation(ArticleCacheID);
    END;
    """)

    conn.commit()
    conn.close()


def fetch_articles(
    sql_config: SqlConfig,
    database: str,
    limit: int,
    namespace: str,
    resume: bool,
):
    conn = connect(sql_config, database)
    cur = conn.cursor()

    namespace_clause = ""
    if namespace != "all":
        namespace_clause = f"AND src.NamespaceID = {int(namespace)}"

    already_translated_clause = ""
    if resume:
        already_translated_clause = """
        AND NOT EXISTS
        (
            SELECT 1
            FROM dbo.WikipediaArticleEnglishTranslation t
            WHERE t.ArticleCacheID = src.ArticleCacheID
        )
        """

    cur.execute(f"""
    SELECT TOP ({int(limit)})
        src.ArticleCacheID,
        src.PageID,
        src.Title,
        src.TitleNormalized,
        src.NamespaceID,
        src.CleanText,
        src.CleanCharCount
    FROM dbo.WikipediaArticleCache src
    WHERE src.CleanText IS NOT NULL
      AND src.CleanCharCount > 0
      {namespace_clause}
      {already_translated_clause}
    ORDER BY src.ArticleCacheID;
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


def clean_for_translation(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def select_translation_excerpt(text: str, max_chars: int) -> str:
    text = clean_for_translation(text)

    if len(text) <= max_chars:
        return text

    # Prefer complete sentence-ish boundaries.
    cut = text[:max_chars]
    boundary = max(
        cut.rfind("।"),
        cut.rfind("."),
        cut.rfind("!"),
        cut.rfind("?"),
        cut.rfind("।"),
        cut.rfind(" "),
    )

    if boundary > max_chars * 0.65:
        return cut[:boundary].strip()

    return cut.strip()


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    text = clean_for_translation(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(part), max_chars):
                chunks.append(part[i:i + max_chars])
            continue

        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def translate_text(translator, text: str, chunk_chars: int) -> str:
    chunks = chunk_text(text, max_chars=chunk_chars)
    translated = []

    for chunk in chunks:
        result = translator(chunk, max_length=512)
        translated.append(result[0]["translation_text"])

    return "\n\n".join(translated).strip()


def insert_translation_row(
    sql_config: SqlConfig,
    database: str,
    article_cache_id: int,
    page_id: Optional[int],
    thai_title: str,
    thai_title_normalized: str,
    english_title_normalized: Optional[str],
    thai_excerpt: Optional[str],
    english_translation: Optional[str],
    source_clean_char_count: int,
    translated_thai_char_count: int,
    translator_model: str,
    status: str,
    error_message: Optional[str],
) -> None:
    conn = connect(sql_config, database)
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO dbo.WikipediaArticleEnglishTranslation
    (
        ArticleCacheID,
        PageID,
        ThaiTitle,
        ThaiTitleNormalized,
        EnglishTitleNormalized,
        ThaiCleanTextExcerpt,
        EnglishCleanTextTranslation,
        SourceCleanCharCount,
        TranslatedThaiCharCount,
        EnglishCharCount,
        TranslatorModel,
        TranslationStatus,
        ErrorMessage
    )
    SELECT
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    WHERE NOT EXISTS
    (
        SELECT 1
        FROM dbo.WikipediaArticleEnglishTranslation
        WHERE ArticleCacheID = ?
    );
    """,
    article_cache_id,
    page_id,
    thai_title,
    thai_title_normalized,
    english_title_normalized,
    thai_excerpt,
    english_translation,
    source_clean_char_count,
    translated_thai_char_count,
    len(english_translation or ""),
    translator_model,
    status,
    error_message,
    article_cache_id)

    conn.commit()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate Thai WikipediaArticleCache fields into English."
    )

    parser.add_argument("--database", default=DEFAULT_SOURCE_DATABASE)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--namespace", default="0", help="'0' for articles, or 'all'.")
    parser.add_argument("--max-cleantext-chars", type=int, default=2500)
    parser.add_argument("--chunk-chars", type=int, default=700)
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-th-en")
    parser.add_argument("--device", type=int, default=-1, help="-1 CPU, 0 first CUDA device.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    sql_config = SqlConfig.from_environment()

    print("=" * 100)
    print("Thai Wikipedia SQL Translation Test")
    print("=" * 100)
    print(f"Database              : {args.database}")
    print(f"Source table          : dbo.WikipediaArticleCache")
    print(f"Output table          : dbo.WikipediaArticleEnglishTranslation")
    print(f"Limit                 : {args.limit}")
    print(f"Namespace             : {args.namespace}")
    print(f"Max CleanText chars   : {args.max_cleantext_chars}")
    print(f"Chunk chars           : {args.chunk_chars}")
    print(f"Translator model      : {args.model}")
    print(f"Device                : {args.device}")
    print(f"Resume                : {args.resume}")
    print(f"Dry run               : {args.dry_run}")
    print("=" * 100)

    if not args.dry_run:
        ensure_translation_table(sql_config, args.database)

    print("Loading translator...")
    translator = pipeline(
        "translation",
        model=args.model,
        device=args.device,
    )
    print("Translator loaded.")

    rows = fetch_articles(
        sql_config=sql_config,
        database=args.database,
        limit=args.limit,
        namespace=args.namespace,
        resume=args.resume,
    )

    print(f"Rows selected: {len(rows)}")

    started = time.time()
    ok_count = 0
    fail_count = 0

    for index, row in enumerate(rows, start=1):
        article_cache_id = int(row.ArticleCacheID)
        page_id = row.PageID
        thai_title = row.Title or ""
        thai_title_normalized = row.TitleNormalized or ""
        clean_text = row.CleanText or ""
        clean_char_count = int(row.CleanCharCount or 0)

        thai_excerpt = select_translation_excerpt(
            clean_text,
            max_chars=args.max_cleantext_chars,
        )

        print("\n" + "-" * 100)
        print(f"{index}/{len(rows)} ArticleCacheID={article_cache_id} PageID={page_id}")
        print(f"Thai title: {thai_title}")
        print(f"Thai excerpt chars: {len(thai_excerpt)}")

        try:
            english_title_normalized = translate_text(
                translator,
                thai_title_normalized,
                chunk_chars=args.chunk_chars,
            )

            english_cleantext = translate_text(
                translator,
                thai_excerpt,
                chunk_chars=args.chunk_chars,
            )

            status = "OK"
            error_message = None
            ok_count += 1

            print(f"English title: {english_title_normalized[:200]}")
            print(f"English text preview: {english_cleantext[:300]}")

        except Exception as exc:
            english_title_normalized = None
            english_cleantext = None
            status = "FAILED"
            error_message = f"{type(exc).__name__}: {exc}"
            fail_count += 1
            print(f"FAILED: {error_message}")

        if not args.dry_run:
            insert_translation_row(
                sql_config=sql_config,
                database=args.database,
                article_cache_id=article_cache_id,
                page_id=page_id,
                thai_title=thai_title,
                thai_title_normalized=thai_title_normalized,
                english_title_normalized=english_title_normalized,
                thai_excerpt=thai_excerpt,
                english_translation=english_cleantext,
                source_clean_char_count=clean_char_count,
                translated_thai_char_count=len(thai_excerpt),
                translator_model=args.model,
                status=status,
                error_message=error_message,
            )

    elapsed = time.time() - started

    print("\n" + "=" * 100)
    print("Translation test complete")
    print("=" * 100)
    print(f"Rows attempted : {len(rows)}")
    print(f"OK             : {ok_count}")
    print(f"FAILED         : {fail_count}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())