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
DEFAULT_INPUT_FILE = "germanInput.txt"

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    input_file = sys.argv[1]
    print(f"Using command-line input file: {input_file}")
else:
    input_file = DEFAULT_INPUT_FILE
    print(f"Using default input file: {input_file}")


# ------------------------------------------------------------
# Load German -> English translator
# ------------------------------------------------------------
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-de-en",
    device=-1,      # CPU
)


# ------------------------------------------------------------
# Read German text file
# ------------------------------------------------------------
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()


print()
print("=" * 80)
print("German -> English Translation")
print("=" * 80)


# ------------------------------------------------------------
# Translate each non-empty line
# ------------------------------------------------------------
for line_number, line in enumerate(lines, start=1):

    german_text = line.strip()

    if not german_text:
        continue

    english_text = translator(
        german_text,
        max_length=400
    )[0]["translation_text"]

    print()
    print("-" * 80)
    print(f"Line {line_number}")

    print("German:")
    print(german_text)

    print()
    print("English:")
    print(english_text)


print()
print("=" * 80)
print("Translation Complete")
print("=" * 80)


# ------------------------------------------------------------
# Alphabetical German Vocabulary Translation
# ------------------------------------------------------------
print()
print("=" * 80)
print("Alphabetical German Vocabulary List")
print("=" * 80)


# Read complete file again
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()


# Extract German words, including umlauts and ß
words = re.findall(
    r"[A-Za-zÄÖÜäöüß'-]+",
    full_text
)


# Normalize for alphabetical vocabulary list
normalized_words = [
    w.lower()
    for w in words
]


# Remove duplicates and sort alphabetically
unique_words = sorted(set(normalized_words))


print()
print(f"Unique German words found: {len(unique_words)}")
print()

print(
    f"{'German Word':30s} "
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
# German Word-by-Word Translation in Original Text Order
# ------------------------------------------------------------
print()
print("=" * 100)
print("German Word-by-Word Translation in Original Text Order")
print("=" * 100)


# Extract words in original order
ordered_words = re.findall(
    r"[A-Za-zÄÖÜäöüß'-]+",
    full_text
)


print()
print(f"Total German word tokens found: {len(ordered_words)}")
print()

print(
    f"{'Position':>8s} "
    f"{'German Word':30s} "
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
print("German Word-by-Word Translation Complete")
print("=" * 100)