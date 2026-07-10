import argparse
from pathlib import Path
import re
from typing import List, Optional

"""

python3 split_classes.py \
  --input_file "./input/Bond.decompiled.cs" \
  --output_dir "./output"

  
"""


class ClassBlock:
    def __init__(self, name: str, start_line: int, parent: Optional[str]):
        self.name = name
        self.start_line = start_line
        self.end_line: Optional[int] = None
        self.parent = parent
        self.start_depth: Optional[int] = None


def find_classes(lines: List[str]) -> List[ClassBlock]:
    brace_depth = 0
    classes: List[ClassBlock] = []
    stack: List[ClassBlock] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # crude skip of comment-only lines
        if stripped.startswith("//"):
            delta_open = line.count("{")
            delta_close = line.count("}")
            brace_depth += delta_open - delta_close
            continue

        # detect class declaration
        match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if match:
            name = match.group(1)
            parent = stack[-1].name if stack else None
            cb = ClassBlock(name=name, start_line=i, parent=parent)
            cb.start_depth = brace_depth
            classes.append(cb)
            stack.append(cb)

        delta_open = line.count("{")
        delta_close = line.count("}")
        brace_depth += delta_open - delta_close

        # close classes when depth drops below their start depth
        while stack and brace_depth < stack[-1].start_depth:
            stack[-1].end_line = i
            stack.pop()

    # any still-open classes end at last line
    last_line = len(lines) - 1
    for cb in classes:
        if cb.end_line is None:
            cb.end_line = last_line

    return classes


def write_class_files(classes: List[ClassBlock], lines: List[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for cb in classes:
        class_lines = lines[cb.start_line:cb.end_line + 1]
        # full class (including nested)
        full_path = output_dir / f"{cb.name}.cs"
        full_path.write_text("".join(class_lines), encoding="utf-8")


def write_denuded_file(classes: List[ClassBlock], lines: List[str], output_dir: Path, original_name: str):
    # remove all class blocks from the original file
    remove_ranges = [(cb.start_line, cb.end_line) for cb in classes]
    denuded_lines = []
    for i, line in enumerate(lines):
        if any(start <= i <= end for start, end in remove_ranges):
            continue
        denuded_lines.append(line)

    denuded_path = output_dir / f"{original_name}.denuded.cs"
    denuded_path.write_text("".join(denuded_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Split C# file into class mini-files and denuded file")
    parser.add_argument("--input_file", required=True, help="Path to the C# source file")
    parser.add_argument("--output_dir", required=True, help="Directory to write generated files")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        return

    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    classes = find_classes(lines)

    if not classes:
        print("No classes found in input file.")
        return

    write_class_files(classes, lines, output_dir)
    write_denuded_file(classes, lines, output_dir, input_path.stem)

    print(f"Processed {len(classes)} classes.")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
