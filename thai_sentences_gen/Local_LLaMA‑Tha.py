
# https://huggingface.co/google/gemma-2-2b-it


import os
import random
import pyodbc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------------
# SQL CONNECTION (Ubuntu)
# ------------------------------------------------------------
DEFAULT_DATABASE = "ThaiDictionary"

def connect(database: str = DEFAULT_DATABASE):
    server = os.getenv("MSSQL_SERVER")
    username = os.getenv("MSSQL_USERNAME")
    password = os.getenv("MSSQL_PASSWORD")

    server="10.0.0.20,63451"
    database = DEFAULT_DATABASE
    username="cubbins"
    password="Hazard123!"


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

# ------------------------------------------------------------
# LOAD THAI WORDS
# ------------------------------------------------------------
def load_thai_words(limit=500):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT TOP (?) ThaiWord
        FROM ThaiDictionary.dbo.ThaiWord
        WHERE ThaiWord IS NOT NULL
        ORDER BY NEWID();
    """, limit)

    return [row[0] for row in cursor.fetchall()]

# ------------------------------------------------------------
# GEMMA 2B INSTRUCT
# ------------------------------------------------------------
MODEL_NAME = "google/gemma-2-2b-it"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# ------------------------------------------------------------
# GENERATE THAI SENTENCE
# ------------------------------------------------------------
def generate_thai_sentence(words, complexity="ง่าย"):
    prompt = (
        f"สร้างประโยคภาษาไทยที่มีความซับซ้อนระดับ: {complexity}\n"
        f"ใช้คำต่อไปนี้อย่างน้อยหนึ่งคำ:\n"
        f"{', '.join(words)}\n"
        f"ประโยคต้องเป็นธรรมชาติและถูกหลักไวยากรณ์:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            temperature=0.8,
            top_p=0.9,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_line_by_line(thai_text):
    lines = thai_text.split("\n")
    translated_pairs = []

    for line in lines:
        cleaned = line.strip()
        if cleaned:
            # Use your model to translate each line
            prompt = f"Translate this Thai sentence into English:\n{cleaned}\nEnglish:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=False
                )

            english = tokenizer.decode(output[0], skip_special_tokens=True)
            english = english.replace(prompt, "").strip()

            translated_pairs.append((cleaned, english))

    return translated_pairs


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    words = load_thai_words(limit=300)
    selected = random.sample(words, 5)

    # Generate all three levels
    simple = generate_thai_sentence(selected, "ง่าย")
    medium = generate_thai_sentence(selected, "ปานกลาง")
    advanced = generate_thai_sentence(selected, "ซับซ้อน")

    print("Selected words:", selected)
    # print("\nSimple:", simple)
    # print("\nMedium:", medium)
    # print("\nAdvanced:", advanced)

    # ------------------------------------------------------------
    # SAVE FULL GENERATIVE OUTPUT TO FILE
    # ------------------------------------------------------------
    output_path = "thai_full_output.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Selected Words ===\n")
        f.write(", ".join(selected) + "\n\n")

        f.write("=== Simple ===\n")
        f.write(simple + "\n\n")

        f.write("=== Medium ===\n")
        f.write(medium + "\n\n")

        f.write("=== Advanced ===\n")
        f.write(advanced + "\n\n")

    print(f"\nFull Thai generative output saved to: {output_path}")

    # ------------------------------------------------------------
    # SAVE FULL THAI + LINE-BY-LINE ENGLISH TRANSLATION
    # ------------------------------------------------------------
    output_path = "thai_full_output_with_translation.txt"

    translated = translate_line_by_line(advanced)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Selected Words ===\n")
        f.write(", ".join(selected) + "\n\n")

        f.write("=== Advanced Thai Output ===\n")
        f.write(advanced + "\n\n")

        f.write("=== Line-by-Line Translation ===\n")
        for thai_line, eng_line in translated:
            f.write(f"THAI: {thai_line}\n")
            f.write(f"ENGLISH: {eng_line}\n\n")

    print(f"\nSaved Thai + English translation to: {output_path}")




if __name__ == "__main__":
    main()
