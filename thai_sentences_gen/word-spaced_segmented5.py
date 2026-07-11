#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import json
import pyodbc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

DEFAULT_DATABASE = "ThaiDictionary"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# OUTPUT_FILE = "thai_sentence_report.txt"

OUTPUT_DIRECTORY = "reports"


LEVELS = ["simple", "intermediate", "advanced"]
WORD_SPACING = "     "

def create_output_filename():
    """
    Create a timestamped report filename.

    Example:
        reports/thai_sentence_report_20260705_193412.txt
    """

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"thai_sentence_report_{timestamp}.txt"
    )

    return os.path.join(OUTPUT_DIRECTORY, filename)




def connect(database: str = DEFAULT_DATABASE):
    server = os.getenv("MSSQL_SERVER", "10.0.0.20,63451")
    username = os.getenv("MSSQL_USERNAME", "cubbins")
    password = os.getenv("MSSQL_PASSWORD")

    if not password:
        raise RuntimeError("Missing MSSQL_PASSWORD environment variable.")

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


def clean_thai_sentence(text: str) -> str:
    text = text.strip()
    text = text.replace("。", ".")
    text = re.sub(r"\s+", "", text)
    return text


def contains_thai(text: str) -> bool:
    return bool(re.search(r"[\u0E00-\u0E7F]", text))


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


def fetch_dictionary_words(max_len: int = 30) -> set[str]:
    conn = connect()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT ThaiWord
        FROM dbo.ThaiWord
        WHERE ThaiWord IS NOT NULL
          AND LEN(ThaiWord) BETWEEN 1 AND ?
        """,
        max_len,
    )

    words = set()

    for row in cur.fetchall():
        word = str(row[0]).strip()
        if is_valid_thai_word(word):
            words.add(word)

    conn.close()
    return words


def fetch_definition_map(words: list[str]) -> dict[str, str]:
    """
    Retrieve definitions from SQL.

    This version does NOT require SenseNo.
    """

    if not words:
        return {}

    conn = connect()
    cur = conn.cursor()

    definition_map = {}

    for word in words:
        cur.execute(
            """
            SELECT TOP (1)
                D.DefinitionText
            FROM dbo.ThaiWord AS W
            INNER JOIN dbo.ThaiDefinition AS D
                ON W.WordID = D.WordID
            WHERE W.ThaiWord = ?
              AND D.DefinitionText IS NOT NULL
            ORDER BY D.DefinitionID;
            """,
            word,
        )

        row = cur.fetchone()

        if row:
            definition = str(row[0]).strip()
            definition = re.sub(r"\s+", " ", definition)
            definition_map[word] = definition

    conn.close()
    return definition_map


def fallback_longest_match_segment(sentence: str, dictionary_words: set[str]) -> list[str]:
    tokens = []
    i = 0
    max_word_len = max((len(w) for w in dictionary_words), default=1)

    while i < len(sentence):
        char = sentence[i]

        if char.isspace():
            i += 1
            continue

        if not re.match(r"[\u0E00-\u0E7F]", char):
            tokens.append(char)
            i += 1
            continue

        best = None
        upper = min(len(sentence), i + max_word_len)

        for j in range(upper, i, -1):
            candidate = sentence[i:j]
            if candidate in dictionary_words:
                best = candidate
                break

        if best:
            tokens.append(best)
            i += len(best)
        else:
            tokens.append(char)
            i += 1

    return [token for token in tokens if token.strip()]


def segment_thai_sentence(sentence: str, dictionary_words: set[str]) -> list[str]:
    sentence = sentence.strip()

    try:
        from pythainlp.tokenize import word_tokenize

        tokens = word_tokenize(
            sentence,
            engine="newmm",
            keep_whitespace=False,
        )

        tokens = [token.strip() for token in tokens if token.strip()]

        if len(tokens) > 1:
            return tokens

    except Exception:
        pass

    return fallback_longest_match_segment(sentence, dictionary_words)


def make_wide_spaced_sentence(tokens: list[str]) -> str:
    return WORD_SPACING.join(tokens)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def ask_model(tokenizer, model, prompt: str, max_new_tokens: int = 600) -> str:
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

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


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
You are a Thai language sentence generator.

Generate ONE natural Thai sentence using at least ONE of the selected Thai words.

Selected Thai words:
{", ".join(words)}

Sentence complexity level:
{level}

Return ONLY valid JSON.

Required JSON format:
{{
  "level": "{level}",
  "used_words": ["word1"],
  "thai_sentence": "Thai sentence here"
}}

Rules:
- The Thai sentence must be grammatical and natural.
- Use at least one selected Thai word.
- Do not include English in the Thai sentence.
- Do not include explanations.
"""

    raw = ask_model(tokenizer, model, prompt)
    data = extract_json(raw)

    if "thai_sentence" not in data:
        raise ValueError(f"Missing thai_sentence key:\n{raw}")

    thai_sentence = clean_thai_sentence(str(data["thai_sentence"]))

    return {
        "level": str(data.get("level", level)).strip(),
        "used_words": data.get("used_words", []),
        "thai_sentence": thai_sentence,
        "raw_output": raw,
    }


def translate_sentence_to_english(tokenizer, model, thai_sentence: str) -> str:
    prompt = f"""
Translate this Thai sentence into natural English.

Thai sentence:
{thai_sentence}

Return ONLY valid JSON.

Required JSON format:
{{
  "english_translation": "English sentence here"
}}

Rules:
- The English translation must contain English only.
- Do not copy Thai words into the English translation.
- Do not explain.
"""

    raw = ask_model(tokenizer, model, prompt, max_new_tokens=400)
    data = extract_json(raw)

    english = str(data.get("english_translation", "")).strip()

    if contains_thai(english) or not english:
        english = "(translation failed; model returned Thai text)"

    return english


def translate_missing_words_with_model(tokenizer, model, missing_words: list[str]) -> dict[str, str]:
    if not missing_words:
        return {}

    prompt = f"""
You are a Thai-English dictionary assistant.

Translate each Thai word into a short English meaning.

Thai words:
{json.dumps(missing_words, ensure_ascii=False)}

Return ONLY valid JSON.

Required JSON format:
{{
  "translations": [
    {{
      "thai": "Thai word",
      "english": "short English meaning"
    }}
  ]
}}

Rules:
- Translate individual words only.
- Keep meanings short.
- English field must contain English only.
- Preserve each Thai word exactly.
"""

    raw = ask_model(tokenizer, model, prompt, max_new_tokens=800)
    data = extract_json(raw)

    result = {}

    for item in data.get("translations", []):
        thai = str(item.get("thai", "")).strip()
        english = str(item.get("english", "")).strip()

        if thai and english and not contains_thai(english):
            result[thai] = english

    return result

def fetch_word_info_map22(words: list[str]) -> dict[str, dict]:
    """
    Retrieve dictionary information for Thai words:
    English definition, IPA, Romanization, ThaiReading, TonePattern.
    """

    if not words:
        return {}

    conn = connect()
    cur = conn.cursor()

    info_map = {}

    for word in words:
        cur.execute(
            """
            SELECT TOP (1)
                W.ThaiWord,
                W.IPA,
                W.Romanization,
                W.ThaiReading,
                W.TonePattern,
                D.DefinitionText
            FROM dbo.ThaiWord AS W
            LEFT JOIN dbo.ThaiDefinition AS D
                ON W.WordID = D.WordID
            WHERE W.ThaiWord = ?
            ORDER BY W.WordID, D.DefinitionID;
            """,
            word,
        )

        row = cur.fetchone()

        if row:
            definition = str(row.DefinitionText or "").strip()
            definition = re.sub(r"\s+", " ", definition)

            if len(definition) > 120:
                definition = definition[:120].rstrip() + "..."

            info_map[word] = {
                "thai": str(row.ThaiWord or "").strip(),
                "ipa": str(row.IPA or "").strip(),
                "romanization": str(row.Romanization or "").strip(),
                "thai_reading": str(row.ThaiReading or "").strip(),
                "tone_pattern": str(row.TonePattern or "").strip(),
                "english": definition or "(no SQL definition)",
            }

    conn.close()
    return info_map

def fetch_word_info_map(words: list[str]) -> dict[str, dict]:
    """
    Retrieve complete lexical information for each Thai word.
    """

    if not words:
        return {}

    conn = connect()
    cur = conn.cursor()

    info = {}

    for word in words:

        cur.execute(
            """
            SELECT TOP (1)
                W.ThaiWord,
                W.IPA,
                W.Romanization,
                W.ThaiReading,
                W.TonePattern,
                W.Etymology,
                D.DefinitionText
            FROM dbo.ThaiWord AS W
            LEFT JOIN dbo.ThaiDefinition AS D
                ON W.WordID = D.WordID
            WHERE W.ThaiWord = ?
            ORDER BY W.WordID, D.DefinitionID;
            """,
            word,
        )

        row = cur.fetchone()

        if row is None:
            continue

        definition = str(row.DefinitionText or "").strip()

        definition = re.sub(r"\s+", " ", definition)

        #
        # Shorten extremely long Wiktionary definitions.
        #
        if len(definition) > 100:
            definition = definition[:100].rstrip() + "..."

        etymology = str(row.Etymology or "").strip()

        if len(etymology) > 80:
            etymology = etymology[:80].rstrip() + "..."

        info[word] = {

            "thai": row.ThaiWord,

            "ipa": row.IPA or "",

            "romanization": row.Romanization or "",

            "thai_reading": row.ThaiReading or "",

            "tone_pattern": row.TonePattern or "",

            "etymology": etymology,

            "english": definition

        }

    conn.close()

    return info



def build_word_breakdown22(tokenizer, model, tokens: list[str]) -> list[dict]:
    unique_tokens = []

    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    sql_defs = fetch_definition_map(unique_tokens)

    missing = [
        token for token in unique_tokens
        if token not in sql_defs and re.search(r"[\u0E00-\u0E7F]", token)
    ]

    model_defs = translate_missing_words_with_model(tokenizer, model, missing)

    glossary = []

    for token in tokens:
        english = sql_defs.get(token) or model_defs.get(token) or "(not translated)"

        glossary.append(
            {
                "thai": token,
                "english": english,
            }
        )

    return glossary

def build_word_breakdown(tokenizer, model, tokens: list[str]) -> list[dict]:
    unique_tokens = []

    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    sql_info = fetch_word_info_map(unique_tokens)

    missing = [
        token for token in unique_tokens
        if token not in sql_info and re.search(r"[\u0E00-\u0E7F]", token)
    ]

    model_defs = translate_missing_words_with_model(tokenizer, model, missing)

    glossary = []

    for token in tokens:
        if token in sql_info:
            item = sql_info[token]
        else:
            item = {
                "thai": token,
                "ipa": "",
                "romanization": "",
                "thai_reading": "",
                "tone_pattern": "",
                "english": model_defs.get(token, "(not translated)"),
            }

        glossary.append(item)

    return glossary



def enrich_results(
    tokenizer,
    model,
    results: list[dict],
    dictionary_words: set[str],
) -> list[dict]:

    for item in results:
        item["english_translation"] = translate_sentence_to_english(
            tokenizer,
            model,
            item["thai_sentence"],
        )

        tokens = segment_thai_sentence(
            item["thai_sentence"],
            dictionary_words=dictionary_words,
        )

        item["thai_tokens"] = tokens
        item["thai_wide_spaced"] = make_wide_spaced_sentence(tokens)
        item["word_breakdown"] = build_word_breakdown(tokenizer, model, tokens)

    return results





def write_report(words, results, filename=None):

    if filename is None:
        filename = create_output_filename()

    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Thai Sentence Generation Report ===\n\n")



        f.write("Selected SQL Words:\n")
        for word in words:
            f.write(f"- {word}\n")

        f.write("\n")


        selected_word_info = fetch_word_info_map(words)

        f.write("=" * 70 + "\n")
        f.write("Selected SQL Words\n")
        f.write("=" * 70 + "\n\n")

        for i, word in enumerate(words, start=1):

            info = selected_word_info.get(word, {})

            f.write(f"Word {i}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Thai Word     : {word}\n")
            f.write(f"IPA           : {info.get('ipa','')}\n")
            f.write(f"Romanization  : {info.get('romanization','')}\n")
            f.write(f"Tone Pattern  : {info.get('tone_pattern','')}\n")
            f.write(f"Etymology     : {info.get('etymology','')}\n")
            f.write(f"Meaning       : {info.get('english','')}\n")
            f.write("\n")



        for index, item in enumerate(results, start=1):
            f.write("=" * 70 + "\n")
            f.write(f"Sentence {index}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Level: {item['level']}\n\n")

            f.write("Used SQL Words:\n")
            for word in item["used_words"]:
                f.write(f"- {word}\n")

            f.write("\n")

            f.write("THAI Standard:\n")
            f.write(f"{item['thai_sentence']}\n\n")

            f.write("THAI Word-Spaced Wide:\n")
            f.write(f"{item['thai_wide_spaced']}\n\n")

            f.write("ENGLISH Sentence Translation:\n")
            f.write(f"{item['english_translation']}\n\n")

            f.write("Word-by-Word Breakdown:\n")
            f.write("=" * 70 + "\n\n")




            for word_index, pair in enumerate(item["word_breakdown"], start=1):
                thai = pair.get("thai", "")
                ipa = pair.get("ipa", "")
                romanization = pair.get("romanization", "")
                tone = pair.get("tone_pattern", "")
                etymology = pair.get("etymology", "")
                english = pair.get("english", "")

                f.write(f"Word {word_index}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Thai Word     : {thai}\n")
                f.write(f"IPA           : {ipa}\n")
                f.write(f"Romanization  : {romanization}\n")
                f.write(f"Tone Pattern  : {tone}\n")
                f.write(f"Etymology     : {etymology}\n")
                f.write(f"Meaning       : {english}\n")
                f.write("\n")

            f.write("ENGLISH Sentence Translation:\n")
            f.write(f"{item['english_translation']}\n\n")



def main():
    print("Retrieving Thai words from SQL Server...")
    words = fetch_words(limit=5)

    print("\nSelected SQL words:")
    for word in words:
        print(f"- {word}")

    print("\nLoading SQL dictionary words for fallback segmentation...")
    dictionary_words = fetch_dictionary_words()
    print(f"Dictionary words loaded: {len(dictionary_words)}")

    print("\nLoading model...")
    tokenizer, model = load_model()

    results = []

    for level in LEVELS:
        print(f"\nGenerating {level} sentence...")

        result = generate_sentence(
            tokenizer=tokenizer,
            model=model,
            words=words,
            level=level,
        )

        results.append(result)

        print("\nTHAI Standard:")
        print(result["thai_sentence"])

    print("\nBuilding translations, segmentation, and glossary...")
    results = enrich_results(
        tokenizer=tokenizer,
        model=model,
        results=results,
        dictionary_words=dictionary_words,
    )

    for item in results:
        print("\nTHAI Word-Spaced Wide:")
        print(item["thai_wide_spaced"])

        print("\nENGLISH Translation:")
        print(item["english_translation"])

    write_report(words, results)

    output_file = create_output_filename()

    write_report(
        words,
        results,
        output_file,
    )

    print("\nReport written to:")
    print(output_file)



if __name__ == "__main__":
    main()