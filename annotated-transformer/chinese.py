import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline
import jieba

# ------------------------------------------------------------
# Load Chinese -> English translator
# ------------------------------------------------------------
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    device=-1,      # CPU
)

# ------------------------------------------------------------
# Read Chinese text file
# ------------------------------------------------------------
input_file = "chineseInput.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print()
print("=" * 80)
print("Chinese -> English Translation")
print("=" * 80)

# ------------------------------------------------------------
# Translate each non-empty line
# ------------------------------------------------------------
for line_number, line in enumerate(lines, start=1):

    chinese_text = line.strip()

    if not chinese_text:
        continue

    english_text = translator(
        chinese_text,
        max_length=256
    )[0]["translation_text"]

    print()
    print("-" * 80)
    print(f"Line {line_number}")

    print("Chinese:")
    print(chinese_text)

    print()
    print("English:")
    print(english_text)

print()
print("=" * 80)
print("Translation Complete")
print("=" * 80)

# ------------------------------------------------------------
# Chinese Vocabulary Translation
# ------------------------------------------------------------
print()
print("=" * 80)
print("Alphabetical Chinese Vocabulary List")
print("=" * 80)

with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# ------------------------------------------------------------
# Chinese word segmentation
# ------------------------------------------------------------
words = jieba.lcut(full_text)

words = [
    w.strip()
    for w in words
    if w.strip()
]

# Remove duplicates
unique_words = sorted(set(words))

print()
print(f"Unique Chinese words found: {len(unique_words)}")
print()

print(f"{'Chinese Word':30s} {'English Translation'}")
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