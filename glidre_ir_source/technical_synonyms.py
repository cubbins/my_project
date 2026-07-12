# technical_synonyms.py

from nltk.corpus import wordnet as wn
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")


def get_synonyms(word):
    synonyms = set()

    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)

    return sorted(synonyms)


def main():
    technical_words = [
        "alliance",
        "competition",
        "security",
        "coercion",
        "deterrence",
        "capability",
        "escalation",
        "sovereignty"
    ]

    for word in technical_words:
        print(f"\n{word}")
        print("-" * len(word))

        synonyms = get_synonyms(word)

        if synonyms:
            for s in synonyms[:20]:
                print(" ", s)
        else:
            print("  No synonyms found.")


if __name__ == "__main__":
    main()