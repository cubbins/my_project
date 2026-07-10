import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


VALID_USINGS = {
    "using System;",
    "using System.Collections.Generic;",
    "using System.Linq;",
    "using System.Linq.Expressions;",
    "using System.Text;",
    "using System.Threading.Tasks;"
}

"""
python3 qwen_standalone_minimal_fixed.py \
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
    return f"""
You are an expert C# engineer.

Your task:
Given the following C# class definition, output TWO SECTIONS ONLY:

USING:
List ONLY the required using statements needed to compile this class.
One per line.

INSTANTIATION:
Provide ONE valid constructor-based instantiation line for this class.
Use correct parameter types.
If unsure, provide a SAFE fallback:
var obj = new ClassName(default, state => Expression.Empty());

DO NOT output JSON.
DO NOT add commentary.
DO NOT modify the class source.

Format EXACTLY:

USING:
using ...

INSTANTIATION:
var obj = new ClassName(...);

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
    return tokenizer.decode(output[0], skip_special_tokens=True)


def parse_sections(text: str):
    using_lines = []
    inst_line = ""

    mode = None
    for raw in text.splitlines():
        line = raw.strip()

        if line.startswith("USING:"):
            mode = "using"
            continue
        if line.startswith("INSTANTIATION:"):
            mode = "inst"
            continue

        if mode == "using" and line.startswith("using "):
            using_lines.append(line)

        if mode == "inst" and line.startswith("var obj"):
            inst_line = line

    return using_lines, inst_line


def sanitize_usings(using_lines):
    cleaned = set()

    for u in using_lines:
        if u in VALID_USINGS:
            cleaned.add(u)

    # Always include System
    cleaned.add("using System;")

    return sorted(cleaned)


def sanitize_instantiation(class_name: str, inst_line: str):
    if not inst_line or "Literal" in inst_line:
        # Fallback safe instantiation
        return f"var obj = new {class_name}(default, state => Expression.Empty());"

    return inst_line


def extract_class_name(class_source: str):
    for line in class_source.splitlines():
        line = line.strip()
        if line.startswith("class ") or " class " in line:
            parts = line.replace("{", "").split()
            for i, p in enumerate(parts):
                if p == "class":
                    return parts[i + 1]
    return "UnknownClass"


def build_standalone_file(class_source: str, using_lines, inst_line):
    using_block = "\n".join(using_lines) + "\n\n"

    main_block = f"""
public static class Program
{{
    public static void Main(string[] args)
    {{
        {inst_line}
    }}
}}
""".strip()

    return using_block + class_source + "\n\n" + main_block + "\n"


def main():
    parser = argparse.ArgumentParser(description="Robust Qwen standalone C# file builder (no JSON)")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen model...")
    tokenizer, model = load_model(args.qwen_model)

    cs_files = list(input_dir.glob("*.cs"))
    print(f"Found {len(cs_files)} class-only C# files.\n")

    for cs_file in cs_files:
        print(f"Processing {cs_file.name}...")
        class_source = cs_file.read_text(encoding="utf-8")

        class_name = extract_class_name(class_source)

        prompt = build_prompt(class_source)
        raw = query_qwen(tokenizer, model, prompt)

        using_lines, inst_line = parse_sections(raw)

        using_lines = sanitize_usings(using_lines)
        inst_line = sanitize_instantiation(class_name, inst_line)

        standalone = build_standalone_file(class_source, using_lines, inst_line)

        out_path = output_dir / f"{cs_file.stem}.standalone.cs"
        out_path.write_text(standalone, encoding="utf-8")

        print(f"Written: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
