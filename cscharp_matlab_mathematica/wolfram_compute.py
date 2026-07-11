from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

# C:\Program Files\Wolfram Research\Wolfram\15.0
kernel_path = r"C:\Program Files\Wolfram Research\Wolfram\15.0\WolframKernel.exe"

print("Starting external Mathematica kernel session...")

with WolframLanguageSession(kernel_path) as session:
    
    # Example A: Pass a raw string equation using wlexpr
    mathematica_string = "Solve[{x + y == 10, x - y == 2}, {x, y}]"
    
    print("\nEvaluating raw string expression...")
    result_string = session.evaluate(wlexpr(mathematica_string))
    print(f"Result from string: {result_string}")
    
    # Example B: Clean way to pass the economic numeric equations
    # Using wlexpr allows you to pass standard text strings instead of complex wl.Times objects
    print("\nEvaluating numeric economics expression...")
    
    econ_query = "Solve[{V0 == 0.95*((1 - 0.2)*V0 + 0.2*(Vz - 0.15*10)), Vz == 10 + 0.95*V0}, {V0, Vz}]"
    
    result_econ = session.evaluate(wlexpr(econ_query))
    print(f"Result from economics system: {result_econ}")
    
    # Optional Example C: If you REALLY want to build it using pure Python factory tokens (wl):
    # Notice you must explicitly declare wl.Times and wl.Plus operations
    print("\nEvaluating pure factory token (wl) expression...")
    
    eq1 = wl.Equal(wl.V0, wl.Times(0.95, wl.Plus(wl.Times(wl.Subtract(1, 0.2), wl.V0), wl.Times(0.2, wl.Subtract(wl.Vz, wl.Times(0.15, 10))))))
    eq2 = wl.Equal(wl.Vz, wl.Plus(10, wl.Times(0.95, wl.V0)))
    structured_query = wl.Solve([eq1, eq2], [wl.V0, wl.Vz])
    
    result_structured = session.evaluate(structured_query)
    print(f"Result from factory token query: {result_structured}")

print("\nSession closed successfully.")


# Assume 'result_econ' is your output: ((Rule[Global`V0, 27.142857142857142], Rule[Global`Vz, 35.785714285714285]),)

def parse_wolfram_rules(wolfram_output):
    result_dict = {}
    # Extract the inner rules from the nested tuples
    if wolfram_output and len(wolfram_output) > 0:
        rules = wolfram_output[0]
        for rule in rules:
            # A Wolfram Rule has a head name 'Rule', a left-hand side, and a right-hand side
            # rule[0] is the Symbol name, rule[1] is the actual numeric value
            symbol_name = str(rule[0]).split('`')[-1] # strips the 'Global`' prefix if present
            value = float(rule[1])
            result_dict[symbol_name] = value
    return result_dict

# Parse the data
clean_data = parse_wolfram_rules(result_econ)

print("--- Parsed Clean Data ---")
print(f"Parsed Dictionary: {clean_data}")
print(f"Extracted V0 directly: {clean_data['V0']}")
print(f"Extracted Vz directly: {clean_data['Vz']}")





