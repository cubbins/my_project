import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )
    return tokenizer, model


def build_analysis_prompt(filename: str, source_code: str) -> str:
    return f"""
You are an expert C# software engineer and code analyst.

Your task is to analyze the following C# source file and produce
a structured, intelligent explanation of the code.

File name:
{filename}

C# source code:
-------------------------
{source_code}
-------------------------

Analysis requirements:
1. Provide a high-level summary of what the file does.
2. List all classes, interfaces, enums, and structs found.
3. List all methods with short descriptions of their purpose.
4. Describe the control flow and important logic.
5. Identify any potential bugs, risks, or inefficiencies.
6. Suggest improvements or refactoring ideas.
7. Comment on naming, structure, and clarity.
8. Explain how the file fits into a typical C# project architecture.

Output format:
- Section 1: Summary
- Section 2: Types Found
- Section 3: Methods
- Section 4: Logic & Behavior
- Section 5: Issues & Risks
- Section 6: Suggested Improvements
- Section 7: Architectural Commentary

Write clearly, intelligently, and with strong reasoning.
Do not invent APIs or behavior not present in the code.
""".strip()


def analyze_cs_file(path: Path, tokenizer, model) -> str:
    source_code = path.read_text(encoding="utf-8")
    prompt = build_analysis_prompt(path.name, source_code)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=1200,
        temperature=0.2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def write_output(output_dir: Path, filename: str, analysis: str):
    output_path = output_dir / f"{filename}.analysis.txt"
    output_path.write_text(analysis, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="C# Code Analyzer using Qwen")
    parser.add_argument("--input_dir", required=True, help="Directory containing .cs files")
    parser.add_argument("--output_dir", required=True, help="Directory to write analysis results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen model...")
    tokenizer, model = load_model()

    cs_files = list(input_dir.glob("*.cs"))
    if not cs_files:
        print("No .cs files found in input directory.")
        return

    print(f"Found {len(cs_files)} C# files. Beginning analysis...\n")

    for cs_file in cs_files:
        print(f"=== Analyzing {cs_file.name} ===")
        analysis = analyze_cs_file(cs_file, tokenizer, model)
        write_output(output_dir, cs_file.stem, analysis)
        print(f"Written analysis to: {output_dir / (cs_file.stem + '.analysis.txt')}\n")


if __name__ == "__main__":
    main()
