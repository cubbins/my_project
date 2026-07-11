#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from datetime import datetime
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

INPUT_FILE = "inputText.txt"
#OUTPUT_FILE = "outputText.txt"
OUTPUT_DIRECTORY = "."

def make_output_filename() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return str(Path(OUTPUT_DIRECTORY) / f"outputText_{timestamp}.txt")


def load_text(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_text(filename: str, text: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")



def save_report(filename: str,
                original_text: str,
                rewritten_text: str,
                model_name: str) -> None:

    with open(filename, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("Qwen Simplification Report\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Model           : {model_name}\n")
        f.write(f"Date/Time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File      : {INPUT_FILE}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("ORIGINAL INPUT TEXT\n")
        f.write("=" * 80 + "\n\n")

        f.write(original_text)
        f.write("\n\n")

        f.write("=" * 80 + "\n")
        f.write("SIMPLIFIED OUTPUT TEXT\n")
        f.write("=" * 80 + "\n\n")

        f.write(rewritten_text.strip())
        f.write("\n")


def build_prompt(source_text: str) -> str:
    return f"""
Restate the following public text in a simplified way.

Instructions:
- Keep the meaning close to the original.
- Use simpler wording.
- Do not add new facts.
- Do not give legal advice.
- It is acceptable if some sentences closely resemble the original.
- Output only the simplified rewritten text.

TEXT:
{source_text}
"""


def main():
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

    output_filename = make_output_filename()

    save_report(
        output_filename,
        source_text,
        simplified_text,
        MODEL_ID
    )

    print(f"Saved report to {output_filename}")
if __name__ == "__main__":
    main()