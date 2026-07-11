import sympy as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

# 1. Path to your local Wolfram Kernel
kernel_path = r"C:\Program Files\Wolfram Research\Mathematica\14.0\WolframKernel.exe"



# C:\Program Files\Wolfram Research\Wolfram\15.0
kernel_path = r"C:\Program Files\Wolfram Research\Wolfram\15.0\WolframKernel.exe"




print("Opening Mathematica session for analytical extraction...")

with WolframLanguageSession(kernel_path) as session:
    # Use standard text queries to solve the math parameters completely in the kernel
    # By mapping strings to the rule values, the output comes back to Python as text rules!
    query = """
    With[{sol = First[Solve[{V0 == beta*((1 - theta)*V0 + theta*(Vz - tau*pz)), Vz == pz + beta*V0}, {V0, Vz}]]},
        {"V0" -> ToString[InputForm[V0 /. sol]], "Vz" -> ToString[InputForm[Vz /. sol]]}
    ]
    """
    print("Computing symbolic equation roots in Mathematica...")
    raw_rules = session.evaluate(wlexpr(query))

# 2. Convert the returned Wolfram rules directly into a clean Python dictionary
# The library automatically translates native Wolfram {"A" -> 1} structures into a python tuple of Rules
parsed_data = {}
for rule in raw_rules:
    # rule[0] is the string key ("V0" or "Vz"), rule[1] is the math string
    parsed_data[str(rule[0])] = str(rule[1])

# 3. Explicitly define symbols to override the SymPy 'beta' function name conflict
V0, Vz, beta, theta, tau, pz = sp.symbols('V0 Vz beta theta tau pz')
local_dict = {
    'V0': V0, 'Vz': Vz, 'beta': beta, 'theta': theta, 'tau': tau, 'pz': pz
}

print("\n--- Parsed Into Native SymPy Mathematical Objects ---")
extracted_solutions = {}

for var_name, expr_str in parsed_data.items():
    # Passing local_dict overrides SymPy's built-in beta function mapping safely
    parsed_expr = sp.sympify(expr_str, locals=local_dict)
    extracted_solutions[var_name] = parsed_expr
    print(f"{var_name} = {parsed_expr}")

print("\n--- Testing Extracted Objects (Simplifying V0) ---")
# Fully manipulable live SymPy mathematical objects
simplified_V0 = sp.simplify(extracted_solutions['V0'])
print(f"Simplified V0 in Python: {simplified_V0}")
