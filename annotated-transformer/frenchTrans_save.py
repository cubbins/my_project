import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline

# ------------------------------------------------------------
# Load French -> English translator
# ------------------------------------------------------------
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-fr-en",
    device=-1,      # CPU
)

# ------------------------------------------------------------
# Read French text file
# ------------------------------------------------------------
input_file = "frenchInput.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print()
print("=" * 80)
print("French -> English Translation")
print("=" * 80)

# ------------------------------------------------------------
# Translate each non-empty line
# ------------------------------------------------------------
for line_number, line in enumerate(lines, start=1):

    french_text = line.strip()

    if not french_text:
        continue

    english_text = translator(
        french_text,
        max_length=256
    )[0]["translation_text"]

    print()
    print("-" * 80)
    print(f"Line {line_number}")
    print("French :")
    print(french_text)
    print()
    print("English:")
    print(english_text)

print()
print("=" * 80)
print("Translation Complete")
print("=" * 80)

# ------------------------------------------------------------
# Alphabetical French Vocabulary Translation
# ------------------------------------------------------------
import re

print()
print("=" * 80)
print("Alphabetical French Vocabulary List")
print("=" * 80)

# Read complete file again
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Extract words
words = re.findall(r"[A-Za-zÀ-ÿ'-]+", full_text)

# Normalize
words = [w.lower() for w in words]

# Remove duplicates
unique_words = sorted(set(words))

print(f"\nUnique French words found: {len(unique_words)}\n")

print(f"{'French Word':30s} {'English Translation'}")
print("-" * 80)

for word in unique_words:

    try:
        translation = translator(
            word,
            max_length=32
        )[0]["translation_text"]

    except Exception as e:
        translation = f"[ERROR: {e}]"

    print(f"{word:30s} {translation}")

print()
print("=" * 80)
print("Vocabulary Translation Complete")
print("=" * 80)


# ------------------------------------------------------------
# French Word-by-Word Translation in Original Text Order
# ------------------------------------------------------------

print()
print("=" * 100)
print("French Word-by-Word Translation in Original Text Order")
print("=" * 100)

# Re-read complete file
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Extract words in original order
ordered_words = re.findall(
    r"[A-Za-zÀ-ÿ'-]+",
    full_text
)

print()
print(f"Total French word tokens found: {len(ordered_words)}")
print()

print(
    f"{'Position':>8s} "
    f"{'French Word':30s} "
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
print("French Word-by-Word Translation Complete")
print("=" * 100)