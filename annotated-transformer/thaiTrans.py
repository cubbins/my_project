import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline

# ------------------------------------------------------------
# Load Thai -> English translator
# ------------------------------------------------------------
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-th-en",
    device=-1,      # CPU
)

# ------------------------------------------------------------
# Read Thai text file
# ------------------------------------------------------------
input_file = "thaiInput.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Split into paragraphs separated by blank lines
paragraphs = [
    p.strip()
    for p in text.split("\n\n")
    if p.strip()
]

print()
print("=" * 80)
print("Thai -> English Translation")
print("=" * 80)

# ------------------------------------------------------------
# Translate each paragraph
# ------------------------------------------------------------
for paragraph_number, thai_text in enumerate(paragraphs, start=1):

    #english_text = translator(
    #    thai_text,
    #    max_length=512
    #)[0]["translation_text"]

    #english_text = translator(
    #    thai_text,
    #    max_length=len(thai_text) * 2
    #)[0]["translation_text"]

    english_text = translator(
        thai_text,
        max_length=2048
    )[0]["translation_text"]

    print()
    print("=" * 80)
    print(f"Paragraph {paragraph_number}")
    print("=" * 80)

    print("\nThai:\n")
    print(thai_text)

    print("\nEnglish:\n")
    print(english_text)

print()
print("=" * 80)
print("Translation Complete")
print("=" * 80)

# ------------------------------------------------------------
# Thai Vocabulary Translation
# ------------------------------------------------------------
from pythainlp.tokenize import word_tokenize

print()
print("=" * 80)
print("Alphabetical Thai Vocabulary List")
print("=" * 80)

# Re-read original file
with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# Thai word segmentation
thai_words = word_tokenize(
    full_text,
    engine="newmm"
)

# Remove whitespace-only tokens
thai_words = [
    w.strip()
    for w in thai_words
    if w.strip()
]

# Remove duplicates
unique_words = sorted(set(thai_words))

print()
print(f"Unique Thai words found: {len(unique_words)}")
print()

# Batch translation (much faster)
translations = translator(
    unique_words,
    max_length=64
)

print(f"{'Thai Word':30s} {'English Translation'}")
print("-" * 80)

for thai_word, result in zip(unique_words, translations):

    english_word = result["translation_text"]

    print(
        f"{thai_word:30s} "
        f"{english_word}"
    )

print()
print("=" * 80)
print("Thai Vocabulary Translation Complete")
print("=" * 80)


# ------------------------------------------------------------
# Thai Word-by-Word Translation in Original Text Order
# ------------------------------------------------------------
from pythainlp.tokenize import word_tokenize

print()
print("=" * 100)
print("Thai Word-by-Word Translation in Original Text Order")
print("=" * 100)

with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

thai_words_in_order = word_tokenize(
    full_text,
    engine="newmm"
)

thai_words_in_order = [
    w.strip()
    for w in thai_words_in_order
    if w.strip()
]

print(f"Total Thai word tokens found: {len(thai_words_in_order)}")
print()

translations = translator(
    thai_words_in_order,
    max_length=64
)

print(
    f"{'Position':>8s} "
    f"{'Thai Word':30s} "
    f"{'English Translation'}"
)
print("-" * 100)

for position, (thai_word, result) in enumerate(
    zip(thai_words_in_order, translations),
    start=1
):
    print(
        f"{position:8d} "
        f"{thai_word:30s} "
        f"{result['translation_text']}"
    )

print()
print("=" * 100)
print("Thai Word-by-Word Translation Complete")
print("=" * 100)

