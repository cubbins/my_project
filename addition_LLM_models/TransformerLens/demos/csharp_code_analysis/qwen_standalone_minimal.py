import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

"""
python3 qwen_standalone_minimal.py \
    --input_dir "./output" \
    --output_dir "./output_standalone"


"""


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )
    return tokenizer, model


def build_prompt(class_source: str):
    """
    Minimal Qwen prompt: infer using statements and constructor instantiation.
    """
    return f"""
You are an expert C# engineer.

Your task:
Given the following C# class definition, produce a JSON structure describing:

- required using statements (only those needed to compile)
- constructor-based instantiation line (valid C#)

Do NOT modify the class source code.
Do NOT add methods.
Do NOT add test scaffolding.

Output JSON format:
{{
  "using_statements": ["using System;", ...],
  "instantiation": "var obj = new ClassName(...);"
}}

Class source:
-------------------------
{class_source}
-------------------------
""".strip()


def query_qwen(tokenizer, model, prompt: str, max_tokens: int = 800):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return json.loads(text)


def build_standalone_file(class_source: str, info: dict) -> str:
    using_block = "\n".join(info["using_statements"]) + "\n\n"
    instantiation = info["instantiation"]

    main_lines = []
    main_lines.append("public static class Program")
    main_lines.append("{")
    main_lines.append("    public static void Main(string[] args)")
    main_lines.append("    {")
    main_lines.append(f"        {instantiation}")
    main_lines.append("    }")
    main_lines.append("}")
    main_block = "\n".join(main_lines)

    return using_block + class_source + "\n\n" + main_block + "\n"


def main():
    parser = argparse.ArgumentParser(description="Minimal Qwen standalone C# file builder")
    parser.add_argument("--input_dir", required=True, help="Directory containing class-only .cs files")
    parser.add_argument("--output_dir", required=True, help="Directory to write standalone .cs files")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct", help="Qwen model name")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen model...")
    tokenizer, model = load_model(args.qwen_model)

    cs_files = list(input_dir.glob("*.cs"))
    if not cs_files:
        print("No .cs files found in input directory.")
        return

    print(f"Found {len(cs_files)} class-only C# files.\n")

    for cs_file in cs_files:
        print(f"Processing {cs_file.name}...")
        class_source = cs_file.read_text(encoding="utf-8")

        prompt = build_prompt(class_source)
        info = query_qwen(tokenizer, model, prompt)

        standalone = build_standalone_file(class_source, info)

        out_path = output_dir / f"{cs_file.stem}.standalone.cs"
        out_path.write_text(standalone, encoding="utf-8")

        print(f"Written: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
