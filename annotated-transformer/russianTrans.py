import sys
import os
import re
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline


# ------------------------------------------------------------
# Choose input file
# ------------------------------------------------------------
DEFAULT_INPUT_FILE = "russianInput.txt"

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    input_file = sys.argv[1]
    print(f"Using command-line input file: {input_file}")
else:
    input_file = DEFAULT_INPUT_FILE
    print(f"Using default input file: {input_file}")


# ------------------------------------------------------------
# Load Russian -> English translator
# ------------------------------------------------------------
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-ru-en",
    device=-1,      # CPU
)


# ------------------------------------------------------------
# Read Russian text file
# ------------------------------------------------------------
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()


print()
print("=" * 80)
print("Russian -> English Translation")
print("=" * 80)


# ------------------------------------------------------------
# Translate each non-empty line
# ------------------------------------------------------------
for line_number, line in enumerate(lines, start=1):

    russian_text = line.strip()

    if not russian_text:
        continue

    english_text = translator(
        russian_text,
        max_length=256
    )[0]["translation_text"]

    print()
    print("-" * 80)
    print(f"Line {line_number}")

    print("Russian:")
    print(russian_text)

    print()
    print("English:")
    print(english_text)


print()
print("=" * 80)
print("Translation Complete")
print("=" * 80)


# ------------------------------------------------------------
# Alphabetical Russian Vocabulary Translation
# ------------------------------------------------------------
print()
print("=" * 80)
print("Alphabetical Russian Vocabulary List")
print("=" * 80)


with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()


# Extract Russian words, including Ё/ё and hyphenated forms
words = re.findall(
    r"[А-Яа-яЁё'-]+",
    full_text
)


normalized_words = [
    w.lower()
    for w in words
]


unique_words = sorted(set(normalized_words))


print()
print(f"Unique Russian words found: {len(unique_words)}")
print()

print(
    f"{'Russian Word':30s} "
    f"{'English Translation'}"
)

print("-" * 80)


for word in unique_words:

    try:
        translation = translator(
            word,
            max_length=32
        )[0]["translation_text"]

    except Exception as e:
        translation = f"[ERROR: {e}]"

    print(
        f"{word:30s} "
        f"{translation}"
    )


print()
print("=" * 80)
print("Vocabulary Translation Complete")
print("=" * 80)


# ------------------------------------------------------------
# Russian Word-by-Word Translation in Original Text Order
# ------------------------------------------------------------
print()
print("=" * 100)
print("Russian Word-by-Word Translation in Original Text Order")
print("=" * 100)


ordered_words = re.findall(
    r"[А-Яа-яЁё'-]+",
    full_text
)


print()
print(f"Total Russian word tokens found: {len(ordered_words)}")
print()

print(
    f"{'Position':>8s} "
    f"{'Russian Word':30s} "
    f"{'English Translation'}"
)

print("-" * 100)


for position, word in enumerate(
    ordered_words,
    start=1
):

    try:
        translation = translator(
            word,
            max_length=32
        )[0]["translation_text"]

    except Exception as e:
        translation = f"[ERROR: {e}]"

    print(
        f"{position:8d} "
        f"{word:30s} "
        f"{translation}"
    )


print()
print("=" * 100)
print("Russian Word-by-Word Translation Complete")
print("=" * 100)