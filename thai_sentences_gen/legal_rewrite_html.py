#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
from html import escape

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

INPUT_FILE = "inputText.txt"
OUTPUT_DIRECTORY = "."


def load_text(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()


def make_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def build_prompt(source_text: str) -> str:
    return f"""
Restate the following public text in a simplified way.

Instructions:
- Keep the meaning close to the original.
- Use simpler wording.
- Do not add new facts.
- Do not give legal advice.
- Output only the simplified rewritten text.

TEXT:
{source_text}
"""


def save_text_report(filename: str, original_text: str, rewritten_text: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Qwen Simplification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model     : {MODEL_ID}\n")
        f.write(f"Date/Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {INPUT_FILE}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ORIGINAL INPUT TEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(original_text + "\n\n")

        f.write("=" * 80 + "\n")
        f.write("SIMPLIFIED OUTPUT TEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(rewritten_text.strip() + "\n")


def save_html_report(filename: str, original_text: str, rewritten_text: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Qwen Simplification Report</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 30px;
        background-color: #f7f7f7;
        color: #222;
    }}

    h1 {{
        text-align: center;
    }}

    .metadata {{
        margin-bottom: 25px;
        padding: 15px;
        background-color: #ffffff;
        border: 1px solid #cccccc;
    }}

    .container {{
        display: flex;
        gap: 20px;
        align-items: stretch;
    }}

    .column {{
        width: 50%;
        background-color: #ffffff;
        border: 1px solid #cccccc;
        padding: 20px;
        box-sizing: border-box;
        white-space: pre-wrap;
        line-height: 1.5;
    }}

    .column h2 {{
        margin-top: 0;
        border-bottom: 1px solid #cccccc;
        padding-bottom: 10px;
    }}
</style>
</head>
<body>

<h1>Qwen Simplification Report</h1>

<div class="metadata">
    <strong>Model:</strong> {escape(MODEL_ID)}<br>
    <strong>Date/Time:</strong> {escape(now)}<br>
    <strong>Input File:</strong> {escape(INPUT_FILE)}
</div>

<div class="container">
    <div class="column">
        <h2>Original Input Text</h2>
{escape(original_text)}
    </div>

    <div class="column">
        <h2>Simplified Output Text</h2>
{escape(rewritten_text.strip())}
    </div>
</div>

</body>
</html>
"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    source_text = load_text(INPUT_FILE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    messages = [
        {
            "role": "user",
            "content": build_prompt(source_text)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    simplified_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    timestamp = make_timestamp()

    text_output_file = Path(OUTPUT_DIRECTORY) / f"outputText_{timestamp}.txt"
    html_output_file = Path(OUTPUT_DIRECTORY) / f"outputText_{timestamp}.html"

    save_text_report(text_output_file, source_text, simplified_text)
    save_html_report(html_output_file, source_text, simplified_text)

    print(f"Saved text report to: {text_output_file}")
    print(f"Saved HTML report to: {html_output_file}")


if __name__ == "__main__":
    main()