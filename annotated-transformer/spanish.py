import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)



from transformers import pipeline

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-es-en",
    device=-1,      # CPU
)

sentences = [
    "Una niña pequeña trepa a una casa de juegos de madera.",
    "Dos hombres jóvenes están afuera.",
    "Un grupo de hombres carga algodón en un camión.",
]

for s in sentences:
    print("Spanish :", s)
    print("English :", translator(s, max_length=72)[0]["translation_text"])
    print()