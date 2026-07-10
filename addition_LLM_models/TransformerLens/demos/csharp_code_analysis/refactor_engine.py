import argparse
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

"""
python3 refactor_engine.py \
  --input_dir "./input" \
  --output_dir "./output" \
  --refactor_level moderate

python3 refactor_engine.py \
  --input_dir "./input" \
  --output_dir "./output" \
  --refactor_level aggressive

python3 refactor_engine.py \
  --input_dir "./input" \
  --output_dir "./output" \
  --refactor_level moderate \
  --gpu_batch \
  --batch_size 2


"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype="auto"
    )
    return tokenizer, model


def get_refactor_params(level: str):
    """
    Controls how aggressive the refactoring is.
    """
    if level == "light":
        return {"max_new_tokens": 600, "temperature": 0.3}
    elif level == "moderate":
        return {"max_new_tokens": 1200, "temperature": 0.25}
    elif level == "aggressive":
        return {"max_new_tokens": 1800, "temperature": 0.2}
    else:
        return {"max_new_tokens": 1200, "temperature": 0.25}


def build_refactor_prompt(filename: str, source_code: str, level: str) -> str:
    """
    Full-file refactoring prompt.
    """
    return f"""
You are an expert C# software engineer.

Your task is to refactor the following C# source file while preserving
its exact behavior. The refactoring level is: {level}.

File name:
{filename}

Original C# source code:
-------------------------
{source_code}
-------------------------

Refactoring requirements:
1. Preserve all behavior and semantics.
2. Improve naming, clarity, and structure.
3. Extract methods where appropriate.
4. Extract classes if responsibilities are mixed.
5. Remove duplication.
6. Simplify conditionals and nested logic.
7. Use modern C# features (pattern matching, switch expressions, LINQ, records).
8. Flatten deeply nested control flow.
9. Improve readability and maintainability.
10. Keep the refactored code valid C#.

Output format:
-------------------------
<refactored_code>
-------------------------

Write only valid C# code. No explanation.
""".strip()


def read_cs_files(input_dir: Path) -> List[Path]:
    return list(input_dir.glob("*.cs"))


def write_output(output_dir: Path, filename: str, refactored_code: str):
    output_path = output_dir / f"{filename}.refactored.cs"
    output_path.write_text(refactored_code, encoding="utf-8")


def refactor_single_file(path: Path, tokenizer, model, level: str) -> str:
    source_code = path.read_text(encoding="utf-8")
    prompt = build_refactor_prompt(path.name, source_code, level)
    params = get_refactor_params(level)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"]
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def refactor_batch(files: List[Path], tokenizer, model, level: str) -> List[str]:
    prompts = []
    for path in files:
        source_code = path.read_text(encoding="utf-8")
        prompts.append(build_refactor_prompt(path.name, source_code, level))

    params = get_refactor_params(level)

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
    parser = argparse.ArgumentParser(description="C# Refactoring Engine using Qwen")
    parser.add_argument("--input_dir", required=True, help="Directory containing .cs files")
    parser.add_argument("--output_dir", required=True, help="Directory to write refactored code")
    parser.add_argument(
        "--refactor_level",
        choices=["light", "moderate", "aggressive"],
        default="moderate",
        help="Control refactoring depth"
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
    print(f"Refactor level: {args.refactor_level}")
    print(f"GPU batch mode: {'ON' if args.gpu_batch else 'OFF'}\n")

    if args.gpu_batch:
        batch_size = max(1, args.batch_size)
        for i in range(0, len(cs_files), batch_size):
            batch_files = cs_files[i:i + batch_size]
            print(f"=== Refactoring batch {i // batch_size + 1} ({len(batch_files)} files) ===")
            refactored_list = refactor_batch(batch_files, tokenizer, model, args.refactor_level)
            for path, refactored_code in zip(batch_files, refactored_list):
                write_output(output_dir, path.stem, refactored_code)
                print(f"Written refactored code to: {output_dir / (path.stem + '.refactored.cs')}")
            print()
    else:
        for cs_file in cs_files:
            print(f"=== Refactoring {cs_file.name} ===")
            refactored_code = refactor_single_file(cs_file, tokenizer, model, args.refactor_level)
            write_output(output_dir, cs_file.stem, refactored_code)
            print(f"Written refactored code to: {output_dir / (cs_file.stem + '.refactored.cs')}\n")


if __name__ == "__main__":
    main()
