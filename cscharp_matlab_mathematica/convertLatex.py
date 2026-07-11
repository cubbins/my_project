import re
from sympy import solve, Symbol
from sympy.parsing.latex import parse_latex

# 1. The original raw LaTeX string
latex_input = r"""\[
\begin{cases}
V^{s}(0, \theta) = β\left[(1 - θ)V^{s}(0, \theta) + θ\left(V^{s}(̅z, \theta) - τp(̅z)\right)\right], \\
V^{s}(̅z, \theta) = p(̅z) + βV^{s}(0, \theta).
\end{cases}
\]"""

def clean_and_parse_latex(latex_str):
    # Remove outer display math tags \[ \] and environment tags
    clean_str = re.sub(r'\\\[|\\\]|\\begin{cases}|\\end{cases}', '', latex_str)
    
    # Fix non-standard characters to standard LaTeX commands
    clean_str = clean_str.replace('β', r'\beta').replace('θ', r'\theta').replace('τ', r'\tau')
    clean_str = clean_str.replace('̅z', 'z_bar') # strip combining overline accent for parser safety
    
    # Split the system into individual equations by the double backslash
    raw_equations = [eq.strip().rstrip(',') for eq in clean_str.split(r'\\') if eq.strip()]
    
    parsed_eqs = []
    for eq in raw_equations:
        # Split left-hand side and right-hand side
        lhs_lat, rhs_lat = eq.split('=')
        
        # Parse into SymPy math objects
        lhs_expr = parse_latex(lhs_lat.strip())
        rhs_expr = parse_latex(rhs_lat.strip())
        
        # Create a SymPy Equation (LHS = RHS is represented as LHS - RHS == 0)
        parsed_eqs.append(lhs_expr - rhs_expr)
        
    return parsed_eqs

# 2. Execute parsing
equations = clean_and_parse_latex(latex_input)

print("--- Parsed SymPy Equations (Set to 0) ---")
for i, eq in enumerate(equations, 1):
    print(f"Equation {i}: {eq} = 0")

# 3. Optional: Define explicit Python symbols to algebraically solve the system
# Let's map the complex parsed functions to simple Python variables to solve for V^s(0, theta) and V^s(z_bar, theta)
V_0 = Symbol('V_s_0')
V_z = Symbol('V_s_z')
beta = Symbol('beta')
theta = Symbol('theta')
tau = Symbol('tau')
p_z = Symbol('p_z')

# Rewrite the parsed logic cleanly in native Python/SymPy syntax:
eq1 = V_0 - (beta * ((1 - theta) * V_0 + theta * (V_z - tau * p_z)))
eq2 = V_z - (p_z + beta * V_0)

# Solve the system of equations for V_0 and V_z
solutions = solve([eq1, eq2], (V_0, V_z))

print("\n--- Algebraic Solutions in Python ---")
print(f"V^s(0, theta) = {solutions[V_0]}")
print(f"V^s(z_bar, theta) = {solutions[V_z]}")
