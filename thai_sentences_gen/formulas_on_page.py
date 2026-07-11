#!/usr/bin/env python3
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = Path("page_13.jpg")
OUTPUT_FILE = Path("page_13_latex.txt")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Local JPG-to-LaTeX Formula Extractor")
    print("=" * 80)
    print(f"Model      : {MODEL_ID}")
    print(f"Image file : {IMAGE_PATH}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Device     : {device}")
    print("=" * 80)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    model.eval()

    print("Opening image...")
    image = Image.open(IMAGE_PATH).convert("RGB")

    prompt = (
        "Extract every mathematical formula from this page image. "
        "Return only clean LaTeX. "
        "Use \\[ ... \\] for displayed equations. "
        "Use inline LaTeX only when the formula appears inline in the text. "
        "Do not explain the page. "
        "Do not include commentary."
    )

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

    print("Preparing model input...")
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    if device == "cuda":
        inputs = inputs.to("cuda")

    print("Generating LaTeX...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    latex_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print("\n--- Extracted LaTeX ---\n")
    print(latex_text)

    OUTPUT_FILE.write_text(latex_text + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"Saved LaTeX output to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()