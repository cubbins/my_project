#!/usr/bin/env python3
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
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1 FROM sys.indexes
        WHERE name = 'UX_WikipediaArticleCache_TitleNormalized'
          AND object_id = OBJECT_ID('dbo.WikipediaArticleCache')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_WikipediaArticleCache_TitleNormalized
        ON dbo.WikipediaArticleCache(TitleNormalized);
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1 FROM sys.indexes
        WHERE name = 'UX_WikipediaArticleCache_PageID'
          AND object_id = OBJECT_ID('dbo.WikipediaArticleCache')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_WikipediaArticleCache_PageID
        ON dbo.WikipediaArticleCache(PageID)
        WHERE PageID IS NOT NULL;
    END;
    """)

    cur.execute("""
    IF OBJECT_ID('dbo.WikipediaImportProgress', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.WikipediaImportProgress
        (
            ImportID INT IDENTITY(1,1) NOT NULL
                CONSTRAINT PK_WikipediaImportProgress PRIMARY KEY,

            DumpFile NVARCHAR(1000) NOT NULL,
            PagesScanned BIGINT NOT NULL DEFAULT 0,
            ArticlesPrepared BIGINT NOT NULL DEFAULT 0,
            ArticlesInserted BIGINT NOT NULL DEFAULT 0,
            LastPageID BIGINT NULL,
            LastTitle NVARCHAR(500) NULL,
            IsComplete BIT NOT NULL DEFAULT 0,
            StartedAt DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
            LastUpdated DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
        );
    END;
    """)

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1 FROM sys.indexes
        WHERE name = 'UX_WikipediaImportProgress_DumpFile'
          AND object_id = OBJECT_ID('dbo.WikipediaImportProgress')
    )
    BEGIN
        CREATE UNIQUE INDEX UX_WikipediaImportProgress_DumpFile
        ON dbo.WikipediaImportProgress(DumpFile);
    END;
    """)

    conn.commit()


def get_or_create_progress(conn, dump_file: Path):
    dump_key = str(dump_file.resolve())

    cur = conn.cursor()

    cur.execute("""
    IF NOT EXISTS
    (
        SELECT 1
        FROM dbo.WikipediaImportProgress
        WHERE DumpFile = ?
    )
    BEGIN
        INSERT INTO dbo.WikipediaImportProgress
            (DumpFile, PagesScanned, ArticlesPrepared, ArticlesInserted)
        VALUES
            (?, 0, 0, 0);
    END;
    """, dump_key, dump_key)

    conn.commit()

    cur.execute("""
    SELECT
        ImportID,
        PagesScanned,
        ArticlesPrepared,
        ArticlesInserted,
        LastPageID,
        LastTitle,
        IsComplete
    FROM dbo.WikipediaImportProgress
    WHERE DumpFile = ?;
    """, dump_key)

    row = cur.fetchone()

    return {
        "ImportID": row.ImportID,
        "PagesScanned": int(row.PagesScanned),
        "ArticlesPrepared": int(row.ArticlesPrepared),
        "ArticlesInserted": int(row.ArticlesInserted),
        "LastPageID": row.LastPageID,
        "LastTitle": row.LastTitle,
        "IsComplete": bool(row.IsComplete),
    }


def update_progress(
    conn,
    import_id: int,
    pages_scanned: int,
    articles_prepared: int,
    articles_inserted: int,
    last_page_id,
    last_title,
    is_complete: bool = False,
):
    cur = conn.cursor()

    cur.execute("""
    UPDATE dbo.WikipediaImportProgress
    SET
        PagesScanned = ?,
        ArticlesPrepared = ?,
        ArticlesInserted = ?,
        LastPageID = ?,
        LastTitle = ?,
        IsComplete = ?,
        LastUpdated = SYSUTCDATETIME()
    WHERE ImportID = ?;
    """,
    pages_scanned,
    articles_prepared,
    articles_inserted,
    last_page_id,
    last_title,
    1 if is_complete else 0,
    import_id)

    conn.commit()


def open_dump(path: Path):
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, "rb")
    return open(path, "rb")


def insert_batch(conn, rows):
    if not rows:
        return 0, 0

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
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """

    inserted = 0
    duplicates = 0

    try:
        cur.executemany(sql, rows)
        conn.commit()
        inserted = len(rows)

    except pyodbc.IntegrityError:
        conn.rollback()

        for row in rows:
            try:
                cur.execute(sql, row)
                conn.commit()
                inserted += 1
            except pyodbc.IntegrityError:
                conn.rollback()
                duplicates += 1

    return inserted, duplicates


def import_dump(
    dump_file: Path,
    database: str,
    max_new_articles: int,
    batch_size: int,
    progress_every: int,
    store_raw_wikitext: bool,
    restart_from_beginning: bool,
):
    conn = connect(database)
    ensure_schema(conn)

    progress = get_or_create_progress(conn, dump_file)

    if progress["IsComplete"] and not restart_from_beginning:
        print("This dump is already marked complete.")
        print("Use --restart-from-beginning if you intentionally want to rescan.")
        conn.close()
        return

    resume_pages_scanned = 0 if restart_from_beginning else progress["PagesScanned"]

    total_pages_scanned = resume_pages_scanned
    total_articles_prepared = 0 if restart_from_beginning else progress["ArticlesPrepared"]
    total_articles_inserted = 0 if restart_from_beginning else progress["ArticlesInserted"]

    run_new_articles = 0
    run_duplicates = 0

    batch = []
    last_page_id = progress["LastPageID"]
    last_title = progress["LastTitle"]

    start_time = time.time()

    print("=" * 100)
    print("Restartable Wikipedia SQL Importer")
    print("=" * 100)
    print(f"Dump file             : {dump_file}")
    print(f"Database              : {database}")
    print(f"Resume pages scanned  : {resume_pages_scanned:,}")
    print(f"Max new articles      : {max_new_articles if max_new_articles else 'no limit'}")
    print(f"Batch size            : {batch_size}")
    print(f"Store raw wikitext    : {store_raw_wikitext}")
    print("=" * 100)

    local_page_counter = 0

    with open_dump(dump_file) as f:
        context = ET.iterparse(f, events=("end",))

        for _event, elem in context:
            if strip_ns(elem.tag) != "page":
                continue

            local_page_counter += 1

            if local_page_counter <= resume_pages_scanned:
                elem.clear()
                continue

            total_pages_scanned += 1

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

                last_page_id = page_id
                last_title = title

                if title and clean_text:
                    raw_to_store = raw_text if store_raw_wikitext else None

                    batch.append(
                        (
                            page_id,
                            title,
                            title_norm,
                            namespace_id,
                            raw_to_store,
                            clean_text,
                            len(raw_text),
                            len(clean_text),
                        )
                    )

                    total_articles_prepared += 1
                    run_new_articles += 1

                if len(batch) >= batch_size:
                    inserted, duplicates = insert_batch(conn, batch)

                    total_articles_inserted += inserted
                    run_duplicates += duplicates

                    batch.clear()

                    update_progress(
                        conn=conn,
                        import_id=progress["ImportID"],
                        pages_scanned=total_pages_scanned,
                        articles_prepared=total_articles_prepared,
                        articles_inserted=total_articles_inserted,
                        last_page_id=last_page_id,
                        last_title=last_title,
                        is_complete=False,
                    )

                if max_new_articles > 0 and run_new_articles >= max_new_articles:
                    break

            elem.clear()

            if progress_every > 0 and total_pages_scanned % progress_every == 0:
                elapsed = time.time() - start_time
                rate = (total_pages_scanned - resume_pages_scanned) / elapsed if elapsed > 0 else 0

                print(
                    f"Pages scanned total: {total_pages_scanned:,} | "
                    f"Articles prepared total: {total_articles_prepared:,} | "
                    f"Articles inserted total: {total_articles_inserted:,} | "
                    f"Duplicates this run: {run_duplicates:,} | "
                    f"Rate this run: {rate:,.0f} pages/sec | "
                    f"Last title: {last_title}"
                )

    if batch:
        inserted, duplicates = insert_batch(conn, batch)

        total_articles_inserted += inserted
        run_duplicates += duplicates

        batch.clear()

    completed = max_new_articles == 0

    update_progress(
        conn=conn,
        import_id=progress["ImportID"],
        pages_scanned=total_pages_scanned,
        articles_prepared=total_articles_prepared,
        articles_inserted=total_articles_inserted,
        last_page_id=last_page_id,
        last_title=last_title,
        is_complete=completed,
    )

    conn.close()

    elapsed = time.time() - start_time

    print("=" * 100)
    print("Run complete")
    print("=" * 100)
    print(f"Pages scanned total       : {total_pages_scanned:,}")
    print(f"Articles prepared total   : {total_articles_prepared:,}")
    print(f"Articles inserted total   : {total_articles_inserted:,}")
    print(f"New articles this run     : {run_new_articles:,}")
    print(f"Duplicates this run       : {run_duplicates:,}")
    print(f"Last page ID              : {last_page_id}")
    print(f"Last title                : {last_title}")
    print(f"Elapsed seconds           : {elapsed:.2f}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump-file", required=True)
    parser.add_argument("--database", default=DEFAULT_DATABASE)

    parser.add_argument(
        "--max-new-articles",
        type=int,
        default=100,
        help="Maximum new namespace-0 articles to process in this run. 0 means no limit.",
    )

    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--store-raw-wikitext", action="store_true")

    parser.add_argument(
        "--restart-from-beginning",
        action="store_true",
        help="Ignore saved progress and rescan from the start. Duplicates are skipped by unique indexes.",
    )

    args = parser.parse_args()

    import_dump(
        dump_file=Path(args.dump_file),
        database=args.database,
        max_new_articles=args.max_new_articles,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        store_raw_wikitext=args.store_raw_wikitext,
        restart_from_beginning=args.restart_from_beginning,
    )


if __name__ == "__main__":
    main()