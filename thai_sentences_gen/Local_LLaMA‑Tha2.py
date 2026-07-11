#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import pyodbc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_DATABASE = "ThaiDictionary"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "thai_sentence_report.txt"

LEVELS = ["simple", "intermediate", "advanced"]


def connect(database: str = DEFAULT_DATABASE):
    server = os.getenv("MSSQL_SERVER")
    username = os.getenv("MSSQL_USERNAME")
    password = os.getenv("MSSQL_PASSWORD")

    server = "10.0.0.20,63451"
    database = DEFAULT_DATABASE
    username = "cubbins"
    password = "Verne123!"

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


def is_valid_thai_word(word: str) -> bool:
    if not word:
        return False

    word = word.strip()

    if len(word) < 2:
        return False

    if re.search(r"[A-Za-z0-9\-\._,;:!?/\\]", word):
        return False

    return bool(re.search(r"[\u0E00-\u0E7F]", word))


def fetch_words(limit: int = 5):
    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT TOP (?)
            W.ThaiWord
        FROM dbo.ThaiWord AS W
        INNER JOIN dbo.ThaiDefinition AS D
            ON W.WordID = D.WordID
        WHERE
            W.ThaiWord IS NOT NULL
            AND D.DefinitionText IS NOT NULL
            AND LEN(W.ThaiWord) BETWEEN 2 AND 20
        ORDER BY NEWID();
        """,
        limit * 10,
    )

    words = []

    for row in cur.fetchall():
        word = str(row[0]).strip()

        if is_valid_thai_word(word) and word not in words:
            words.append(word)

        if len(words) >= limit:
            break

    conn.close()

    if not words:
        raise RuntimeError("No valid Thai words were retrieved from the database.")

    return words


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def ask_model(tokenizer, model, prompt: str, max_new_tokens: int = 400) -> str:
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def extract_json(raw_text: str) -> dict:
    raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in model output:\n{raw_text}")

    return json.loads(match.group(0))


def generate_sentence(tokenizer, model, words, level: str) -> dict:
    prompt = f"""
You are a Thai language sentence generator and translator.

Your task:
Generate ONE natural Thai sentence using at least ONE of the selected Thai words.

Selected Thai words:
{", ".join(words)}

Sentence complexity level:
{level}

Return ONLY valid JSON.
Do not include Markdown.
Do not include explanations.
Do not repeat these instructions.
Do not translate the prompt.
Translate only the generated Thai sentence.

Required JSON format:
{{
  "level": "{level}",
  "used_words": ["word1"],
  "thai_sentence": "Thai sentence here",
  "english_translation": "English translation here"
}}

Rules:
- The Thai sentence must be grammatical and natural.
- The English translation must match the Thai sentence.
- Use at least one selected Thai word.
- Do not include unrelated examples.
"""

    raw = ask_model(tokenizer, model, prompt)
    data = extract_json(raw)

    required = ["level", "used_words", "thai_sentence", "english_translation"]

    for key in required:
        if key not in data:
            raise ValueError(f"Missing required JSON key: {key}\nRaw output:\n{raw}")

    return {
        "level": data["level"],
        "used_words": data["used_words"],
        "thai_sentence": data["thai_sentence"],
        "english_translation": data["english_translation"],
        "raw_output": raw,
    }


def write_report(words, results, filename: str = OUTPUT_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Thai Sentence Generation Report ===\n\n")

        f.write("Selected Words:\n")
        for word in words:
            f.write(f"- {word}\n")

        f.write("\n")

        for index, item in enumerate(results, start=1):
            f.write("=" * 60 + "\n")
            f.write(f"Sentence {index}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Level: {item['level']}\n")
            f.write(f"Used words: {', '.join(item['used_words'])}\n")
            f.write(f"THAI: {item['thai_sentence']}\n")
            f.write(f"ENGLISH: {item['english_translation']}\n\n")


def main():
    print("Retrieving Thai words from SQL Server...")
    words = fetch_words(limit=5)

    print("Selected words:")
    for word in words:
        print(f"- {word}")

    print("\nLoading model...")
    tokenizer, model = load_model()

    results = []

    for level in LEVELS:
        print(f"\nGenerating {level} sentence...")
        result = generate_sentence(tokenizer, model, words, level)
        results.append(result)

        print(f"THAI: {result['thai_sentence']}")
        print(f"ENGLISH: {result['english_translation']}")

    write_report(words, results)

    print(f"\nReport written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()