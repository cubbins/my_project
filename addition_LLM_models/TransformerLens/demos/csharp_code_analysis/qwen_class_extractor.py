import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


"""
python3 qwen_class_extractor.py \
  --input_file "./input/Bond.decompiled.cs" \
   --output_dir "./output_qwen_class_ex" \
    --analysis_depth medium

python3 qwen_class_extractor.py \
   --input_file "./input/Bond.decompiled.cs" \
    --output_dir "./output_qwen_class_ex" \
    --analysis_depth deep



"""

def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )
    return tokenizer, model


def build_qwen_prompt(full_source: str, depth: str):
    """
    Qwen prompt: extract using statements, classes, nested classes,
    methods, and produce structured JSON describing the file.
    """
    return f"""
You are an expert C# code analyst.

Your task:
Scan the FULL C# source code below and produce a JSON structure describing:
- all using statements
- all classes
- all nested classes
- all methods inside each class

Depth level: {depth}

Rules:
- Preserve exact class names.
- Preserve nested class relationships.
- Preserve method names.
- Do NOT rewrite code.
- Do NOT summarize.
- Only extract structure.

Output JSON format:
{{
  "using_statements": ["using System;", ...],
  "classes": [
    {{
      "name": "ClassName",
      "parent": null or "OuterClass",
      "source": "full class text",
      "methods": ["Method1", "Method2", ...]
    }},
    ...
  ]
}}

Now process the following C# file:

-------------------------
{full_source}
-------------------------
""".strip()


def generate_structure(tokenizer, model, prompt: str, max_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def write_class_files(structure: dict, output_dir: Path):
    using_block = "\n".join(structure["using_statements"]) + "\n\n"

    for cls in structure["classes"]:
        name = cls["name"]
        parent = cls["parent"]
        source = cls["source"]
        methods = cls["methods"]

        # Build main method
        main_code = []
        main_code.append("public static void Main(string[] args)")
        main_code.append("{")
        main_code.append(f"    var obj = new {name}();")
        main_code.append("")
        main_code.append("    // Example method calls:")
        for m in methods:
            main_code.append(f"    // obj.{m}();")
        main_code.append("}")
        main_code = "\n".join(main_code)

        full_output = using_block + source + "\n\n" + main_code + "\n"

        out_path = output_dir / f"{name}.cs"
        out_path.write_text(full_output, encoding="utf-8")


def write_denuded_file(full_source: str, structure: dict, output_dir: Path, original_name: str):
    denuded = full_source

    for cls in structure["classes"]:
        denuded = denuded.replace(cls["source"], "")

    out_path = output_dir / f"{original_name}.denuded.cs"
    out_path.write_text(denuded, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Qwen-powered C# class extractor")
    parser.add_argument("--input_file", required=True, help="Path to the C# file")
    parser.add_argument("--output_dir", required=True, help="Directory to write output files")
    parser.add_argument("--qwen_model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Qwen model name")
    parser.add_argument("--analysis_depth", choices=["light", "medium", "deep"],
                        default="medium", help="Controls Qwen output size")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_source = input_path.read_text(encoding="utf-8")

    # Depth → token budget
    depth_tokens = {
        "light": 600,
        "medium": 1200,
        "deep": 2000
    }
    max_tokens = depth_tokens[args.analysis_depth]

    print("Loading Qwen model...")
    tokenizer, model = load_model(args.qwen_model)

    print("Generating structure...")
    prompt = build_qwen_prompt(full_source, args.analysis_depth)
    raw_output = generate_structure(tokenizer, model, prompt, max_tokens)

    # Parse JSON from Qwen output
    import json
    structure = json.loads(raw_output)

    print("Writing class files...")
    write_class_files(structure, output_dir)

    print("Writing denuded file...")
    write_denuded_file(full_source, structure, output_dir, input_path.stem)

    print("Done.")


if __name__ == "__main__":
    main()
