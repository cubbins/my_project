from paddlenlp import Taskflow
from pprint import pprint

schema = [
    "Country",
    "Organization",
    "Person",
    "Conflict",
    "Technology",
    "Economic Sanction",
    "Alliance",
    "Strategic Resource",
    "Export Control",
]

print("Loading UIE...")

uie = Taskflow(
    "information_extraction",
    schema=schema,
    model="uie-base-en"
)

text = """
China announced additional export controls on gallium and germanium.
The United States criticized the decision.
NATO members discussed supply chain resilience during a meeting in Brussels.
"""

print("Running extraction...")
result = uie(text)

pprint(result)