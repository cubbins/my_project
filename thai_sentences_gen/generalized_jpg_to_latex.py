#!/usr/bin/env python3

# python generalized_jpg_to_latex.py page_13.jpg --subject "analog electronics"
# python generalized_jpg_to_latex.py ./pages --subject "statistics"
# python generalized_jpg_to_latex.py ./pages \
#  --model Qwen/Qwen2.5-VL-7B-Instruct \
#  --use-4bit \
#  --subject "economics and statistics"


from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_prompt(subject: str) -> str:
    return f"""
You are an expert technical document transcription system.

The page may be about {subject}, economics, statistics, mathematics,
engineering, analog electronics, or another technical subject.

Transcribe the visible page content into clean LaTeX/Markdown.

Requirements:
1. Preserve all equations exactly.
2. Preserve subscripts and superscripts carefully.
3. Use \\[ ... \\] for displayed equations.
4. Use inline math only for inline formulas.
5. Preserve variable names exactly, such as V_t, V_GS, V_DS, i_D, r_DS.
6. Preserve units exactly.
7. Preserve numbered sections such as (a), (b), (c).
8. For tables, use Markdown tables.
9. For charts or diagrams, briefly describe the visible structure.
10. Do not summarize.
11. Do not add explanations.
12. Return only the transcription.
""".strip()


def load_model(model_id: str, use_4bit: bool):
    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    if use_4bit:
        from transformers import BitsAndBytesConfig

        print("Using 4-bit bitsandbytes quantization")

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
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    model.eval()
    return processor, model


def process_image(
    image_path: Path,
    output_dir: Path,
    processor,
    model,
    subject: str,
    max_new_tokens: int,
) -> Path:
    print(f"\nProcessing: {image_path}")

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

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

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
    return output_file


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        images = []
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            images.extend(input_path.glob(pattern))
        return sorted(images)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="General JPG technical-page transcription to LaTeX/Markdown."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="A JPG file or a directory containing JPG files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("latex_outputs"),
        help="Directory where output files will be saved.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Vision-language model ID.",
    )

    parser.add_argument(
        "--subject",
        default="general technical material",
        help="Optional subject hint, e.g. economics, statistics, analog electronics.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1600,
        help="Maximum number of output tokens.",
    )

    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization. Recommended for Qwen2.5-VL-7B on 8GB GPUs.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("General JPG-to-LaTeX/Markdown Technical OCR")
    print("=" * 80)
    print(f"Model       : {args.model}")
    print(f"Input       : {args.input}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Subject hint: {args.subject}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}")

    print("=" * 80)

    processor, model = load_model(args.model, args.use_4bit)

    image_files = collect_images(args.input)

    if not image_files:
        print("No JPG files found.")
        return

    for image_path in image_files:
        process_image(
            image_path=image_path,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            subject=args.subject,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()