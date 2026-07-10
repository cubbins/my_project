import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline
from pypinyin import pinyin, Style
import jieba

# ------------------------------------------------------------
# Load Chinese -> English translator
# ------------------------------------------------------------

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    device=-1,
)

# ------------------------------------------------------------
# Pinyin helper
# ------------------------------------------------------------

def get_pinyin(text):
    return " ".join(
        syllable[0]
        for syllable in pinyin(
            text,
            style=Style.TONE
        )
    )

# ------------------------------------------------------------
# Read Chinese text file
# ------------------------------------------------------------

input_file = "chineseInput.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# ------------------------------------------------------------
# Split into paragraphs
# ------------------------------------------------------------

paragraphs = [
    p.strip()
    for p in text.split("\n\n")
    if p.strip()
]

print()
print("=" * 100)
print("Chinese -> English Translation")
print("=" * 100)

# ------------------------------------------------------------
# Translate each paragraph
# ------------------------------------------------------------

for paragraph_number, chinese_text in enumerate(paragraphs, start=1):

    english_text = translator(
        chinese_text,
        max_length=2048
    )[0]["translation_text"]

    print()
    print("=" * 100)
    print(f"Paragraph {paragraph_number}")
    print("=" * 100)

    print("\nChinese:\n")
    print(chinese_text)

    print("\nEnglish:\n")
    print(english_text)

print()
print("=" * 100)
print("Translation Complete")
print("=" * 100)

# ------------------------------------------------------------
# Alphabetical Chinese Vocabulary List
# ------------------------------------------------------------

print()
print("=" * 100)
print("Alphabetical Chinese Vocabulary List")
print("=" * 100)

with open(input_file, "r", encoding="utf-8") as f:
    full_text = f.read()

# ------------------------------------------------------------
# Chinese word segmentation
# ------------------------------------------------------------

words = list(jieba.cut(full_text))

words = [
    w.strip()
    for w in words
    if w.strip()
]

unique_words = sorted(set(words))

print()
print(f"Unique Chinese words found: {len(unique_words)}")
print()

print(
    f"{'Chinese Word':20s} "
    f"{'Pinyin':30s} "
    f"{'English Translation'}"
)

print("-" * 120)

for word in unique_words:

    try:

        translation = translator(
            word,
            max_length=32
        )[0]["translation_text"]

    except Exception as e:

        translation = f"[ERROR: {e}]"

    pinyin_text = get_pinyin(word)

    print(
        f"{word:20s} "
        f"{pinyin_text:30s} "
        f"{translation}"
    )

print()
print("=" * 100)
print("Alphabetical Vocabulary Translation Complete")
print("=" * 100)

# ------------------------------------------------------------
# Word-by-word report in original text order
# ------------------------------------------------------------

print()
print("=" * 120)
print("Chinese Word-by-Word Translation in Original Text Order")
print("=" * 120)

ordered_words = list(jieba.cut(full_text))

ordered_words = [
    w.strip()
    for w in ordered_words
    if w.strip()
]

print()
print(f"Total Chinese word tokens found: {len(ordered_words)}")
print()

print(
    f"{'Position':>8s} "
    f"{'Chinese Word':20s} "
    f"{'Pinyin':30s} "
    f"{'English Translation'}"
)

print("-" * 140)

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

    pinyin_text = get_pinyin(word)

    print(
        f"{position:8d} "
        f"{word:20s} "
        f"{pinyin_text:30s} "
        f"{translation}"
    )

print()
print("=" * 120)
print("Word-by-Word Translation Complete")
print("=" * 120)