
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)


from transformers import pipeline

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-fr-en",
    device=-1,
)

sentences = [
    "Une petite fille grimpe dans une maison de jeu en bois.",
    "Deux jeunes hommes sont dehors.",
    "Un groupe d'hommes charge du coton sur un camion.",
]

for s in sentences:
    print("French :", s)
    print("English:", translator(s, max_length=72)[0]["translation_text"])
    print()
