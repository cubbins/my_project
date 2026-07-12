from paddlenlp import Taskflow
from pprint import pprint

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
    "Alliance",
]

uie = Taskflow(
    "information_extraction",
    schema=schema,
    model="uie-base",
    precision="fp32",
    device_id=-1,
    use_fast=False,
)

text = """
China announced additional export controls on gallium and germanium.
The United States criticized the decision.
NATO members discussed supply chain resilience during a meeting in Brussels.
"""

print("\nInput Text:")
print(text)

print("\nExtraction:")
results = uie(text)
pprint(results)