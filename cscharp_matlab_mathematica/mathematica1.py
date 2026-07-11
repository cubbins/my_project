import sympy as sp
from sympy.printing.mathematica import mathematica_code

# 1. Define all symbols explicitly in SymPy
V_0, V_z, beta, theta, tau, p_z = sp.symbols('V_0 V_z beta theta tau p_z')

# 2. Build the exact math equations directly using Python objects.
# This completely eliminates string manipulation, missing asterisk bugs, and eval() crashes.
eq1 = V_0 - (beta * ((1 - theta) * V_0 + theta * (V_z - tau * p_z)))
eq2 = V_z - (p_z + beta * V_0)

# 3. Algebraically solve the system inside Python
solutions = sp.solve([eq1, eq2], (V_0, V_z))

# 4. Generate the exact Wolfram Mathematica strings
print("=== MATHEMATICA CODE GENERATION ===\n")

# Format individual equations into Mathematica syntax
m_eq1 = mathematica_code(eq1)
m_eq2 = mathematica_code(eq2)

print("m_eq1 = {m_eq1}",m_eq1)
print("m_eq2 = {m_eq2}")

print("1. Native Mathematica Code to Solve System:")
print(f"Solve[{{{m_eq1} == 0, {m_eq2} == 0}}, {{V_0, V_z}}]\n")

print("2. Mathematica Formatted Solutions:")
m_sol_V0 = mathematica_code(solutions[V_0])
m_sol_Vz = mathematica_code(solutions[V_z])

print(f"V_0 = {m_sol_V0}")
print(f"V_z = {m_sol_Vz}")
