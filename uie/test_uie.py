from paddlenlp import Taskflow

print("=" * 70)
print("Loading UIE...")
print("=" * 70)

schema = [
    "Country",
    "Organization",
    "Person",
    "Conflict",
    "Technology",
    "Economic Sanction",
    "Alliance"
]

uie = Taskflow(
    "information_extraction",
    schema=schema
)

text = """
China announced additional export controls on gallium and germanium.
The United States criticized the decision.
NATO members discussed supply chain resilience during a meeting in Brussels.
"""

print("\nInput Text\n")
print(text)

print("\nExtraction\n")

results = uie(text)

from pprint import pprint
pprint(results)