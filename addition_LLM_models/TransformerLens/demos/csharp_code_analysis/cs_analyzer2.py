import argparse
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype="auto"
    )
    return tokenizer, model


def get_generation_params(depth: str):
    """
    Control speed vs detail.
    """
    if depth == "shallow":
        return {"max_new_tokens": 400, "temperature": 0.5}
    elif depth == "medium":
        return {"max_new_tokens": 800, "temperature": 0.3}
    elif depth == "deep":
        return {"max_new_tokens": 1400, "temperature": 0.2}
    else:
        return {"max_new_tokens": 800, "temperature": 0.3}


def build_analysis_prompt(filename: str, source_code: str) -> str:
    """
    Full-file analysis prompt (no trimming).
    """
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


def read_cs_files(input_dir: Path) -> List[Path]:
    return list(input_dir.glob("*.cs"))


def write_output(output_dir: Path, filename: str, analysis: str):
    output_path = output_dir / f"{filename}.analysis.txt"
    output_path.write_text(analysis, encoding="utf-8")


def analyze_single_file(path: Path, tokenizer, model, depth: str) -> str:
    source_code = path.read_text(encoding="utf-8")
    prompt = build_analysis_prompt(path.name, source_code)
    params = get_generation_params(depth)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"]
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def analyze_batch(files: List[Path], tokenizer, model, depth: str) -> List[str]:
    prompts = []
    for path in files:
        source_code = path.read_text(encoding="utf-8")
        prompts.append(build_analysis_prompt(path.name, source_code))

    params = get_generation_params(depth)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"]
    )

    results = []
    for i in range(len(files)):
        text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        results.append(text)
    return results


def main():
    parser = argparse.ArgumentParser(description="C# Code Analyzer using Qwen")
    parser.add_argument("--input_dir", required=True, help="Directory containing .cs files")
    parser.add_argument("--output_dir", required=True, help="Directory to write analysis results")
    parser.add_argument(
        "--analysis_depth",
        choices=["shallow", "medium", "deep"],
        default="medium",
        help="Control analysis depth and runtime"
    )
    parser.add_argument(
        "--gpu_batch",
        action="store_true",
        help="Enable GPU batch inference over multiple files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for GPU batch inference"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen model...")
    tokenizer, model = load_model()

    cs_files = read_cs_files(input_dir)
    if not cs_files:
        print("No .cs files found in input directory.")
        return

    print(f"Found {len(cs_files)} C# files.")
    print(f"Analysis depth: {args.analysis_depth}")
    print(f"GPU batch mode: {'ON' if args.gpu_batch else 'OFF'}\n")

    if args.gpu_batch:
        # Batched inference
        batch_size = max(1, args.batch_size)
        for i in range(0, len(cs_files), batch_size):
            batch_files = cs_files[i:i + batch_size]
            print(f"=== Analyzing batch {i // batch_size + 1} ({len(batch_files)} files) ===")
            analyses = analyze_batch(batch_files, tokenizer, model, args.analysis_depth)
            for path, analysis in zip(batch_files, analyses):
                write_output(output_dir, path.stem, analysis)
                print(f"Written analysis to: {output_dir / (path.stem + '.analysis.txt')}")
            print()
    else:
        # Single-file inference
        for cs_file in cs_files:
            print(f"=== Analyzing {cs_file.name} ===")
            analysis = analyze_single_file(cs_file, tokenizer, model, args.analysis_depth)
            write_output(output_dir, cs_file.stem, analysis)
            print(f"Written analysis to: {output_dir / (cs_file.stem + '.analysis.txt')}\n")


if __name__ == "__main__":
    main()
