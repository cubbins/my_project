import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)

from transformers import pipeline
from pythainlp.tokenize import word_tokenize

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
# Helper: translate words with up to 3 candidate English outputs
# ------------------------------------------------------------
def translate_words_with_candidates(words, num_candidates=3):
    """
    Translate each Thai word and request up to num_candidates alternatives.
    Handles both flat and nested Hugging Face pipeline output formats.
    """

    if not words:
        return []

    raw_outputs = translator(
        words,
        max_length=64,
        num_beams=5,
        num_return_sequences=num_candidates,
    )

    grouped_results = []

    # Case 1: output is nested:
    # [
    #   [{'translation_text': '...'}, {'translation_text': '...'}, ...],
    #   [{'translation_text': '...'}, {'translation_text': '...'}, ...],
    # ]
    if raw_outputs and isinstance(raw_outputs[0], list):

        for word, candidate_list in zip(words, raw_outputs):

            candidates = [
                item["translation_text"]
                for item in candidate_list
            ]

            clean_candidates = []
            for c in candidates:
                if c not in clean_candidates:
                    clean_candidates.append(c)

            grouped_results.append((word, clean_candidates))

        return grouped_results

    # Case 2: output is flat:
    # [
    #   {'translation_text': '...'},
    #   {'translation_text': '...'},
    #   {'translation_text': '...'},
    #   ...
    # ]
    for i, word in enumerate(words):

        start = i * num_candidates
        end = start + num_candidates

        candidates = [
            item["translation_text"]
            for item in raw_outputs[start:end]
        ]

        clean_candidates = []
        for c in candidates:
            if c not in clean_candidates:
                clean_candidates.append(c)

        grouped_results.append((word, clean_candidates))

    return grouped_results


# ------------------------------------------------------------
# Thai Vocabulary Translation: Alphabetical Unique Words
# ------------------------------------------------------------
print()
print("=" * 100)
print("Alphabetical Thai Vocabulary List")
print("Up to three English candidates per Thai word")
print("=" * 100)

thai_words = word_tokenize(
    text,
    engine="newmm"
)

thai_words = [
    w.strip()
    for w in thai_words
    if w.strip()
]

unique_words = sorted(set(thai_words))

print()
print(f"Unique Thai words found: {len(unique_words)}")
print()

vocabulary_results = translate_words_with_candidates(
    unique_words,
    num_candidates=3
)

print(
    f"{'Thai Word':30s} "
    f"{'English Candidate 1':25s} "
    f"{'English Candidate 2':25s} "
    f"{'English Candidate 3'}"
)
print("-" * 110)

for thai_word, candidates in vocabulary_results:

    c1 = candidates[0] if len(candidates) > 0 else ""
    c2 = candidates[1] if len(candidates) > 1 else ""
    c3 = candidates[2] if len(candidates) > 2 else ""

    print(
        f"{thai_word:30s} "
        f"{c1:25s} "
        f"{c2:25s} "
        f"{c3}"
    )

print()
print("=" * 100)
print("Thai Vocabulary Translation Complete")
print("=" * 100)


# ------------------------------------------------------------
# Thai Word-by-Word Translation in Original Text Order
# ------------------------------------------------------------
print()
print("=" * 120)
print("Thai Word-by-Word Translation in Original Text Order")
print("Up to three English candidates per Thai word")
print("=" * 120)

thai_words_in_order = word_tokenize(
    text,
    engine="newmm"
)

thai_words_in_order = [
    w.strip()
    for w in thai_words_in_order
    if w.strip()
]

print(f"Total Thai word tokens found: {len(thai_words_in_order)}")
print()

ordered_results = translate_words_with_candidates(
    thai_words_in_order,
    num_candidates=3
)

print(
    f"{'Position':>8s} "
    f"{'Thai Word':30s} "
    f"{'English Candidate 1':25s} "
    f"{'English Candidate 2':25s} "
    f"{'English Candidate 3'}"
)
print("-" * 120)

for position, (thai_word, candidates) in enumerate(
    ordered_results,
    start=1
):

    c1 = candidates[0] if len(candidates) > 0 else ""
    c2 = candidates[1] if len(candidates) > 1 else ""
    c3 = candidates[2] if len(candidates) > 2 else ""

    print(
        f"{position:8d} "
        f"{thai_word:30s} "
        f"{c1:25s} "
        f"{c2:25s} "
        f"{c3}"
    )

print()
print("=" * 120)
print("Thai Word-by-Word Translation Complete")
print("=" * 120)