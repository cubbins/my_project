
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)


from transformers import pipeline

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-th-en",
    device=-1,
)

sentences = [
    "สวัสดีครับ",
    "ผมกำลังเรียนภาษาไทย",
    "วันนี้อากาศดีมาก",
]

for s in sentences:
    print("Thai    :", s)
    print("English :", translator(s)[0]["translation_text"])
    print()