#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
from typing import Iterable

import pyodbc


DEFAULT_DATABASE = "WikipediaSearchDB"


def connect(database: str):
    server = os.getenv("MSSQL_SERVER", "10.0.0.20,63451")
    username = os.getenv("MSSQL_USERNAME", "cubbins")
    password = os.getenv("MSSQL_PASSWORD", "Heavy123!")

    conn_str = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    return pyodbc.connect(conn_str, timeout=30)


def ensure_schema(conn):
    cur = conn.cursor()

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaParagraph', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.WikipediaParagraph
        (
            ParagraphID BIGINT IDENTITY(1,1) NOT NULL
                CONSTRAINT PK_WikipediaParagraph PRIMARY KEY,

            ArticleCacheID BIGINT NOT NULL,
            PageID BIGINT NULL,
            Title NVARCHAR(1000) NOT NULL,
            TitleNormalized NVARCHAR(1000) NOT NULL,
            NamespaceID INT NOT NULL,

            ParagraphNumber INT NOT NULL,
            ParagraphText NVARCHAR(MAX) NOT NULL,
            ParagraphCharCount INT NOT NULL,

            DateCreated DATETIME2 NOT NULL
                CONSTRAINT DF_WikipediaParagraph_DateCreated
                DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1
        FROM sys.indexes
        WHERE name = 'UX_WikipediaParagraph_Article_Paragraph'
          AND object_id = OBJECT_ID('dbo.WikipediaParagraph')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_WikipediaParagraph_Article_Paragraph
        ON dbo.WikipediaParagraph(ArticleCacheID, ParagraphNumber);
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1
        FROM sys.indexes
        WHERE name = 'IX_WikipediaParagraph_PageID'
          AND object_id = OBJECT_ID('dbo.WikipediaParagraph')
    )
    BEGIN
        CREATE INDEX IX_WikipediaParagraph_PageID
        ON dbo.WikipediaParagraph(PageID);
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1
        FROM sys.indexes
        WHERE name = 'IX_WikipediaParagraph_NamespaceID'
          AND object_id = OBJECT_ID('dbo.WikipediaParagraph')
    )
    BEGIN
        CREATE INDEX IX_WikipediaParagraph_NamespaceID
        ON dbo.WikipediaParagraph(NamespaceID);
    END;
    """)

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaParagraphProgress', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.WikipediaParagraphProgress
        (
            ProgressID INT IDENTITY(1,1) NOT NULL
                CONSTRAINT PK_WikipediaParagraphProgress PRIMARY KEY,

            LastArticleCacheID BIGINT NOT NULL DEFAULT 0,
            ArticlesProcessed BIGINT NOT NULL DEFAULT 0,
            ParagraphsInserted BIGINT NOT NULL DEFAULT 0,
            StartedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
            LastUpdated DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        );

        INSERT INTO dbo.WikipediaParagraphProgress
            (LastArticleCacheID, ArticlesProcessed, ParagraphsInserted)
        VALUES
            (0, 0, 0);
    END;
    """)

    conn.commit()


def reset_paragraph_tables(conn):
    cur = conn.cursor()

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaParagraph', 'U') IS NOT NULL
        DROP TABLE dbo.WikipediaParagraph;
    """)

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaParagraphProgress', 'U') IS NOT NULL
        DROP TABLE dbo.WikipediaParagraphProgress;
    """)

    conn.commit()
    ensure_schema(conn)


def get_progress(conn):
    cur = conn.cursor()

    cur.execute("""
    SELECT TOP (1)
        ProgressID,
        LastArticleCacheID,
        ArticlesProcessed,
        ParagraphsInserted
    FROM dbo.WikipediaParagraphProgress
    ORDER BY ProgressID;
    """)

    row = cur.fetchone()

    return {
        "ProgressID": int(row.ProgressID),
        "LastArticleCacheID": int(row.LastArticleCacheID),
        "ArticlesProcessed": int(row.ArticlesProcessed),
        "ParagraphsInserted": int(row.ParagraphsInserted),
    }


def update_progress(
    conn,
    progress_id: int,
    last_article_cache_id: int,
    articles_processed: int,
    paragraphs_inserted: int,
):
    cur = conn.cursor()

    cur.execute("""
    UPDATE dbo.WikipediaParagraphProgress
    SET
        LastArticleCacheID = ?,
        ArticlesProcessed = ?,
        ParagraphsInserted = ?,
        LastUpdated = SYSUTCDATETIME()
    WHERE ProgressID = ?;
    """,
    last_article_cache_id,
    articles_processed,
    paragraphs_inserted,
    progress_id)

    conn.commit()


def split_into_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Prefer blank-line paragraph boundaries.
    parts = re.split(r"\n\s*\n+", text)

    paragraphs: list[str] = []

    for part in parts:
        p = part.strip()
        p = re.sub(r"[ \t]+", " ", p)
        p = re.sub(r"\n+", " ", p)
        p = p.strip()

        if len(p) >= min_chars:
            paragraphs.append(p)

    return paragraphs


def fetch_articles(conn, last_article_cache_id: int, article_batch_size: int):
    cur = conn.cursor()

    cur.execute(f"""
    SELECT TOP ({int(article_batch_size)})
        ArticleCacheID,
        PageID,
        Title,
        TitleNormalized,
        NamespaceID,
        CleanText
    FROM dbo.WikipediaArticleCache
    WHERE ArticleCacheID > ?
    ORDER BY ArticleCacheID;
    """, last_article_cache_id)

    return cur.fetchall()


def insert_paragraphs(conn, rows):
    if not rows:
        return 0, 0

    sql = """
    INSERT INTO dbo.WikipediaParagraph
    (
        ArticleCacheID,
        PageID,
        Title,
        TitleNormalized,
        NamespaceID,
        ParagraphNumber,
        ParagraphText,
        ParagraphCharCount
    )
    SELECT
        ?, ?, ?, ?, ?, ?, ?, ?
    WHERE NOT EXISTS
    (
        SELECT 1
        FROM dbo.WikipediaParagraph WITH (UPDLOCK, HOLDLOCK)
        WHERE ArticleCacheID = ?
          AND ParagraphNumber = ?
    );
    """

    inserted = 0
    duplicates = 0

    cur = conn.cursor()
    cur.fast_executemany = False

    for row in rows:
        (
            article_cache_id,
            page_id,
            title,
            title_normalized,
            namespace_id,
            paragraph_number,
            paragraph_text,
            paragraph_char_count,
        ) = row

        params = (
            article_cache_id,
            page_id,
            title,
            title_normalized,
            namespace_id,
            paragraph_number,
            paragraph_text,
            paragraph_char_count,
            article_cache_id,
            paragraph_number,
        )

        cur.execute(sql, params)

        if cur.rowcount == 1:
            inserted += 1
        else:
            duplicates += 1

    conn.commit()

    return inserted, duplicates


def build_paragraph_table(
    database: str,
    mode: str,
    max_articles: int,
    article_batch_size: int,
    paragraph_insert_batch_size: int,
    min_paragraph_chars: int,
    namespace_filter: str,
):
    conn = connect(database)

    if mode == "reset":
        reset_paragraph_tables(conn)
    else:
        ensure_schema(conn)

    progress = get_progress(conn)

    last_article_cache_id = progress["LastArticleCacheID"]
    articles_processed_total = progress["ArticlesProcessed"]
    paragraphs_inserted_total = progress["ParagraphsInserted"]

    articles_processed_this_run = 0
    paragraphs_inserted_this_run = 0
    duplicates_this_run = 0

    start = time.time()

    print("=" * 90)
    print("Wikipedia Paragraph Table Builder")
    print("=" * 90)
    print(f"Database                    : {database}")
    print(f"Mode                        : {mode}")
    print(f"Resume LastArticleCacheID   : {last_article_cache_id}")
    print(f"Max articles this run       : {max_articles if max_articles else 'no limit'}")
    print(f"Article batch size          : {article_batch_size}")
    print(f"Paragraph insert batch size : {paragraph_insert_batch_size}")
    print(f"Minimum paragraph chars     : {min_paragraph_chars}")
    print(f"Namespace filter            : {namespace_filter}")
    print("=" * 90)

    while True:
        article_rows = fetch_articles(conn, last_article_cache_id, article_batch_size)

        if not article_rows:
            print("No more articles found.")
            break

        paragraph_rows = []

        for article in article_rows:
            article_cache_id = int(article.ArticleCacheID)
            page_id = article.PageID
            title = article.Title
            title_normalized = article.TitleNormalized
            namespace_id = int(article.NamespaceID)
            clean_text = article.CleanText or ""

            if namespace_filter != "all":
                if namespace_id != int(namespace_filter):
                    last_article_cache_id = article_cache_id
                    continue

            paragraphs = split_into_paragraphs(
                clean_text,
                min_chars=min_paragraph_chars,
            )

            for i, paragraph in enumerate(paragraphs, start=1):
                paragraph_rows.append(
                    (
                        article_cache_id,
                        page_id,
                        title,
                        title_normalized,
                        namespace_id,
                        i,
                        paragraph,
                        len(paragraph),
                    )
                )

                if len(paragraph_rows) >= paragraph_insert_batch_size:
                    inserted, duplicates = insert_paragraphs(conn, paragraph_rows)

                    paragraphs_inserted_total += inserted
                    paragraphs_inserted_this_run += inserted
                    duplicates_this_run += duplicates

                    paragraph_rows.clear()

            last_article_cache_id = article_cache_id
            articles_processed_total += 1
            articles_processed_this_run += 1

            if max_articles > 0 and articles_processed_this_run >= max_articles:
                break

        if paragraph_rows:
            inserted, duplicates = insert_paragraphs(conn, paragraph_rows)

            paragraphs_inserted_total += inserted
            paragraphs_inserted_this_run += inserted
            duplicates_this_run += duplicates

            paragraph_rows.clear()

        update_progress(
            conn,
            progress["ProgressID"],
            last_article_cache_id,
            articles_processed_total,
            paragraphs_inserted_total,
        )

        elapsed = time.time() - start
        rate = articles_processed_this_run / elapsed if elapsed > 0 else 0

        print(
            f"LastArticleCacheID={last_article_cache_id:,} | "
            f"Articles this run={articles_processed_this_run:,} | "
            f"Paragraphs this run={paragraphs_inserted_this_run:,} | "
            f"Duplicates this run={duplicates_this_run:,} | "
            f"Rate={rate:,.1f} articles/sec"
        )

        if max_articles > 0 and articles_processed_this_run >= max_articles:
            break

    conn.close()

    elapsed = time.time() - start

    print("=" * 90)
    print("Paragraph build complete")
    print("=" * 90)
    print(f"Articles processed this run : {articles_processed_this_run:,}")
    print(f"Paragraphs inserted this run: {paragraphs_inserted_this_run:,}")
    print(f"Duplicates this run         : {duplicates_this_run:,}")
    print(f"LastArticleCacheID          : {last_article_cache_id:,}")
    print(f"Elapsed seconds             : {elapsed:.2f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--mode", choices=["resume", "reset"], default="resume")

    parser.add_argument("--max-articles", type=int, default=1000)
    parser.add_argument("--article-batch-size", type=int, default=500)
    parser.add_argument("--paragraph-insert-batch-size", type=int, default=1000)
    parser.add_argument("--min-paragraph-chars", type=int, default=40)

    parser.add_argument(
        "--namespace",
        default="all",
        help="Use 'all' for every namespace, or a namespace number such as 0, 4, 10, 14.",
    )

    args = parser.parse_args()

    build_paragraph_table(
        database=args.database,
        mode=args.mode,
        max_articles=args.max_articles,
        article_batch_size=args.article_batch_size,
        paragraph_insert_batch_size=args.paragraph_insert_batch_size,
        min_paragraph_chars=args.min_paragraph_chars,
        namespace_filter=args.namespace,
    )


if __name__ == "__main__":
    main()