import sympy as sp
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
import json
# 1. Path to your local Wolfram Kernel
kernel_path = r"C:\Program Files\Wolfram Research\Mathematica\14.0\WolframKernel.exe"


# C:\Program Files\Wolfram Research\Wolfram\15.0
kernel_path = r"C:\Program Files\Wolfram Research\Wolfram\15.0\WolframKernel.exe"


print("Opening Mathematica session for analytical extraction...")

with WolframLanguageSession(kernel_path) as session:
    # We ask Mathematica to export the solved rules matrix directly as a standard JSON string
    query = """
    ExportString[
      First[Solve[
        {V0 == beta*((1 - theta)*V0 + theta*(Vz - tau*pz)), Vz == pz + beta*V0}, 
        {V0, Vz}
      ]], 
      "JSON"
    ]
    """
    print("Computing symbolic equation roots in Mathematica...")
    json_string_result = session.evaluate(wlexpr(query))

# 2. Parse the clean JSON object directly into a standard Python dictionary
# This results in: {"V0": "-((beta*pz*theta - ...", "Vz": "-((pz - ..."}
raw_data = json.loads(json_string_result)

# 3. Explicitly define symbols to protect the 'beta' function name conflict
V0, Vz, beta, theta, tau, pz = sp.symbols('V0 Vz beta theta tau pz')
local_dict = {
    'V0': V0, 'Vz': Vz, 'beta': beta, 'theta': theta, 'tau': tau, 'pz': pz
}

print("\n--- Parsed Into Native SymPy Mathematical Objects ---")
extracted_solutions = {}

for var_name, expr_str in raw_data.items():
    # Passing local_dict overrides SymPy's built-in beta function mapping safely
    parsed_expr = sp.sympify(expr_str, locals=local_dict)
    extracted_solutions[var_name] = parsed_expr
    print(f"{var_name} = {parsed_expr}")

print("\n--- Testing Extracted Objects (Simplifying V0) ---")
# Fully manipulable live SymPy mathematical objects
simplified_V0 = sp.simplify(extracted_solutions['V0'])
print(f"Simplified V0 in Python: {simplified_V0}")
