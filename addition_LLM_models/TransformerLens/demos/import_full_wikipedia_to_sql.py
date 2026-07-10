#!/usr/bin/env python3
"""
import_full_wikipedia_to_sql.py

Purpose:
    Stream the full Wikipedia XML dump into SQL Server.

First test:
    Insert only 100 namespace-0 articles.

Later:
    Increase --max-articles or set --max-articles 0 for no limit.
"""

from __future__ import annotations

import argparse
import bz2
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pyodbc


DEFAULT_DATABASE = "WikipediaSearchDB"


def normalize_title(title: str) -> str:
    text = unquote(str(title or "")).strip().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text(elem: ET.Element, name: str) -> str:
    for child in elem:
        if strip_ns(child.tag) == name:
            return child.text or ""
    return ""


def revision_text(page_elem: ET.Element) -> str:
    for child in page_elem:
        if strip_ns(child.tag) == "revision":
            for rev_child in child:
                if strip_ns(rev_child.tag) == "text":
                    return rev_child.text or ""
    return ""


def clean_wikitext(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)

    text = re.sub(r"<ref[^>/]*/>", " ", text, flags=re.I)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.S | re.I)

    text = re.sub(r"\[\[(?:File|Image|Category):[^\]]+\]\]", " ", text, flags=re.I)

    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]+\]", " ", text)

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"'{2,}", "", text)

    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.M)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def connect(database: str):
    server = os.getenv("MSSQL_SERVER")
    username = os.getenv("MSSQL_USERNAME")
    password = os.getenv("MSSQL_PASSWORD")

    if not server or not username or not password:
        raise RuntimeError(
            "Missing MSSQL_SERVER, MSSQL_USERNAME, or MSSQL_PASSWORD environment variable."
        )

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

    cur.execute(
        """
        IF OBJECT_ID('dbo.WikipediaArticleCache', 'U') IS NULL
        BEGIN
            CREATE TABLE dbo.WikipediaArticleCache
            (
                ArticleCacheID BIGINT IDENTITY(1,1) NOT NULL
                    CONSTRAINT PK_WikipediaArticleCache PRIMARY KEY,

                PageID BIGINT NULL,
                Title NVARCHAR(500) NOT NULL,
                TitleNormalized NVARCHAR(500) NOT NULL,
                NamespaceID INT NOT NULL,

                RawWikiText NVARCHAR(MAX) NULL,
                CleanText NVARCHAR(MAX) NOT NULL,

                RawCharCount INT NOT NULL,
                CleanCharCount INT NOT NULL,

                DateLoaded DATETIME2 NOT NULL
                    CONSTRAINT DF_WikipediaArticleCache_DateLoaded
                    DEFAULT SYSUTCDATETIME()
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
            WHERE name = 'UX_WikipediaArticleCache_TitleNormalized'
              AND object_id = OBJECT_ID('dbo.WikipediaArticleCache')
        )
        BEGIN
            CREATE UNIQUE INDEX UX_WikipediaArticleCache_TitleNormalized
            ON dbo.WikipediaArticleCache(TitleNormalized);
        END;
        """
    )

    cur.execute(
        """
        IF NOT EXISTS
        (
            SELECT 1
            FROM sys.indexes
            WHERE name = 'IX_WikipediaArticleCache_PageID'
              AND object_id = OBJECT_ID('dbo.WikipediaArticleCache')
        )
        BEGIN
            CREATE INDEX IX_WikipediaArticleCache_PageID
            ON dbo.WikipediaArticleCache(PageID);
        END;
        """
    )

    conn.commit()


def open_dump(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return open(path, "rb")


def insert_batch(conn, rows):
    if not rows:
        return 0

    cur = conn.cursor()
    cur.fast_executemany = True

    sql = """
    INSERT INTO dbo.WikipediaArticleCache
        (
            PageID,
            Title,
            TitleNormalized,
            NamespaceID,
            RawWikiText,
            CleanText,
            RawCharCount,
            CleanCharCount
        )
    SELECT ?, ?, ?, ?, ?, ?, ?, ?
    WHERE NOT EXISTS
    (
        SELECT 1
        FROM dbo.WikipediaArticleCache WITH (UPDLOCK, HOLDLOCK)
        WHERE TitleNormalized = ?
    );
    """

    params = []

    for row in rows:
        params.append(
            (
                row["PageID"],
                row["Title"],
                row["TitleNormalized"],
                row["NamespaceID"],
                row["RawWikiText"],
                row["CleanText"],
                row["RawCharCount"],
                row["CleanCharCount"],
                row["TitleNormalized"],
            )
        )

    cur.executemany(sql, params)
    conn.commit()

    return len(rows)


def import_wikipedia(
    dump_file: Path,
    database: str,
    max_articles: int,
    batch_size: int,
    progress_every: int,
    store_raw_wikitext: bool,
):
    conn = connect(database)
    ensure_schema(conn)

    scanned_pages = 0
    prepared_articles = 0
    inserted_articles = 0
    batch = []

    start = time.time()

    print("=" * 90)
    print("Wikipedia full XML import to SQL Server")
    print("=" * 90)
    print(f"Dump file          : {dump_file}")
    print(f"Database           : {database}")
    print(f"Max articles       : {max_articles if max_articles else 'no limit'}")
    print(f"Batch size         : {batch_size}")
    print(f"Store raw wikitext : {store_raw_wikitext}")
    print("=" * 90)

    with open_dump(dump_file) as f:
        context = ET.iterparse(f, events=("end",))

        for _event, elem in context:
            if strip_ns(elem.tag) != "page":
                continue

            scanned_pages += 1

            title = child_text(elem, "title")
            ns_text = child_text(elem, "ns")
            page_id_text = child_text(elem, "id")

            try:
                namespace_id = int(ns_text)
            except Exception:
                namespace_id = -1

            if namespace_id == 0:
                try:
                    page_id = int(page_id_text)
                except Exception:
                    page_id = None

                raw_text = revision_text(elem)
                clean_text = clean_wikitext(raw_text)
                title_norm = normalize_title(title)

                if title and clean_text:
                    batch.append(
                        {
                            "PageID": page_id,
                            "Title": title,
                            "TitleNormalized": title_norm,
                            "NamespaceID": namespace_id,
                            "RawWikiText": raw_text if store_raw_wikitext else None,
                            "CleanText": clean_text,
                            "RawCharCount": len(raw_text),
                            "CleanCharCount": len(clean_text),
                        }
                    )

                    prepared_articles += 1

                if len(batch) >= batch_size:
                    inserted_articles += insert_batch(conn, batch)
                    batch.clear()

                if max_articles > 0 and prepared_articles >= max_articles:
                    break

            elem.clear()

            if progress_every > 0 and scanned_pages % progress_every == 0:
                elapsed = time.time() - start
                rate = scanned_pages / elapsed if elapsed > 0 else 0
                print(
                    f"Scanned pages: {scanned_pages:,} | "
                    f"Prepared articles: {prepared_articles:,} | "
                    f"Inserted/batched: {inserted_articles:,} | "
                    f"Rate: {rate:,.0f} pages/sec"
                )

    if batch:
        inserted_articles += insert_batch(conn, batch)
        batch.clear()

    conn.close()

    elapsed = time.time() - start

    print("=" * 90)
    print("Import complete")
    print("=" * 90)
    print(f"Pages scanned      : {scanned_pages:,}")
    print(f"Articles prepared  : {prepared_articles:,}")
    print(f"Rows sent to SQL   : {inserted_articles:,}")
    print(f"Elapsed seconds    : {elapsed:.2f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump-file", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--max-articles", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--store-raw-wikitext", action="store_true")

    args = parser.parse_args()

    import_wikipedia(
        dump_file=Path(args.dump_file),
        database=args.database,
        max_articles=args.max_articles,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        store_raw_wikitext=args.store_raw_wikitext,
    )


if __name__ == "__main__":
    main()