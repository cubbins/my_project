#!/usr/bin/env python3

###############################################################################
# Example Command Lines
###############################################################################

"""
===============================================================================
Qwen2.5-VL OCR Examples
===============================================================================

1. Qwen 2.5 VL 3B (BF16)
------------------------

python generalized_qwen_vl_ocr.py page_13.jpg \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --mode bf16 \
    --subject "analog electronics"


2. Qwen 2.5 VL 3B (4-bit Quantized)
-----------------------------------

python generalized_qwen_vl_ocr.py page_13.jpg \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --mode 4bit \
    --subject "analog electronics"


3. Qwen 2.5 VL 7B (4-bit Quantized)
-----------------------------------

python generalized_qwen_vl_ocr.py page_13.jpg \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --mode 4bit \
    --subject "analog electronics"


python generalized_qwen_vl_ocr.py ./jpg_pages \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --mode 4bit \
    --subject "analog electronics"    


4. Process an Entire Directory
------------------------------

python generalized_qwen_vl_ocr.py ./jpg_pages \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --mode 4bit \
    --subject "statistics"

===============================================================================
Available Modes
===============================================================================

bf16    Native bfloat16 inference (recommended if the model fits in GPU memory)

4bit    4-bit NF4 quantization (recommended for limited GPU memory)

8bit    8-bit quantization (optional)

===============================================================================
Recommended Models
===============================================================================

Qwen/Qwen2.5-VL-3B-Instruct
    Fastest. Recommended default.

Qwen/Qwen2.5-VL-7B-Instruct
    Higher accuracy. Use 4-bit mode on 8 GB GPUs.

===============================================================================
"""


from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt(subject: str) -> str:
    return f"""
You are an expert technical document transcription system.

The image may contain material from {subject}, economics, statistics,
mathematics, engineering, analog electronics, or another technical field.

Transcribe the visible page into clean Markdown with LaTeX.

Requirements:
1. Preserve all equations exactly.
2. Preserve subscripts and superscripts carefully.
3. Use \\[ ... \\] for displayed equations.
4. Use inline math only for inline formulas.
5. Preserve variable names exactly.
6. Preserve units exactly.
7. Preserve numbered sections such as (a), (b), (c).
8. For tables, use Markdown tables.
9. For charts or diagrams, briefly describe the visible structure.
10. Do not summarize.
11. Do not add explanations.
12. Return only the transcription.
""".strip()


def load_processor_and_model(
    model_id: str,
    mode: str,
):
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=False,
    )

    cuda_available = torch.cuda.is_available()

    if mode == "bf16":
        print("Loading model in BF16 mode")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if cuda_available else torch.float32,
            device_map="auto" if cuda_available else None,
        )

    elif mode == "4bit":
        print("Loading model in 4-bit NF4 mode")

        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

    elif mode == "8bit":
        print("Loading model in 8-bit mode")

        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    model.eval()
    return processor, model


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files: list[Path] = []
        for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
            files.extend(input_path.glob(pattern))
        return sorted(files)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def process_one_image(
    image_path: Path,
    output_dir: Path,
    processor,
    model,
    subject: str,
    max_new_tokens: int,
) -> Path:
    print("=" * 80)
    print(f"Processing: {image_path}")
    print("=" * 80)

    image = Image.open(image_path).convert("RGB")
    prompt = build_prompt(subject)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")

    start_time = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    elapsed = time.perf_counter() - start_time

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{image_path.stem}_transcription.md"
    output_file.write_text(output_text + "\n", encoding="utf-8")

    print(f"Saved: {output_file}")
    print(f"Generation time: {elapsed:.2f} seconds")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="General JPG technical-page OCR to Markdown/LaTeX using Qwen2.5-VL."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input JPG file or directory containing JPG files.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model ID, for example Qwen/Qwen2.5-VL-3B-Instruct or Qwen/Qwen2.5-VL-7B-Instruct.",
    )

    parser.add_argument(
        "--mode",
        choices=["bf16", "4bit", "8bit"],
        default="bf16",
        help="Model loading mode.",
    )

    parser.add_argument(
        "--subject",
        default="general technical material",
        help="Subject hint, for example: analog electronics, statistics, economics.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen_vl_outputs"),
        help="Directory for output Markdown files.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1800,
        help="Maximum generated output tokens.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Generalized JPG-to-LaTeX/Markdown OCR")
    print("=" * 80)
    print(f"Input        : {args.input}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Model        : {args.model}")
    print(f"Mode         : {args.mode}")
    print(f"Subject      : {args.subject}")
    print(f"CUDA         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime : {torch.version.cuda}")

    print("=" * 80)

    processor, model = load_processor_and_model(
        model_id=args.model,
        mode=args.mode,
    )

    image_files = collect_images(args.input)

    if not image_files:
        print("No JPG files found.")
        return

    print(f"Images found: {len(image_files)}")

    for image_path in image_files:
        process_one_image(
            image_path=image_path,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            subject=args.subject,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()