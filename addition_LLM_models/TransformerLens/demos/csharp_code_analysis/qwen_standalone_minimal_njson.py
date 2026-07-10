import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

""""
python3 qwen_standalone_minimal_njson.py \
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

SECTION USING:
List ONLY the required using statements needed to compile this class.
One per line.

SECTION INSTANTIATION:
Provide ONE valid constructor-based instantiation line for this class.
Use correct parameter types.

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
    for line in text.splitlines():
        line = line.strip()

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

        prompt = build_prompt(class_source)
        raw = query_qwen(tokenizer, model, prompt)

        using_lines, inst_line = parse_sections(raw)

        standalone = build_standalone_file(class_source, using_lines, inst_line)

        out_path = output_dir / f"{cs_file.stem}.standalone.cs"
        out_path.write_text(standalone, encoding="utf-8")

        print(f"Written: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
