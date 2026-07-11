import sympy as sp

# 1. Define all symbols explicitly
V_0, V_z, beta, theta, tau, p_z = sp.symbols('V_0 V_z beta theta tau p_z')

# 2. Build the equations
eq1 = V_0 - (beta * ((1 - theta) * V_0 + theta * (V_z - tau * p_z)))
eq2 = V_z - (p_z + beta * V_0)

# 3. Create a dictionary of your numeric values
# Change these values to match your specific economic model parameters
numeric_values = {
    beta: 0.95,    # Discount factor
    theta: 0.20,   # Probability parameter
    tau: 0.15,     # Tax / friction parameter
    p_z: 10.0      # Production / value function parameter at z_bar
}

print("=== METHOD 1: Substituting into the analytical solutions ===")
# Solve algebraically first, then substitute numbers
analytical_solutions = sp.solve([eq1, eq2], (V_0, V_z))

numeric_V_0 = analytical_solutions[V_0].subs(numeric_values)
numeric_V_z = analytical_solutions[V_z].subs(numeric_values)

# Use .evalf() to force floating-point decimal output
print(f"V^s(0, theta)     = {numeric_V_0.evalf():.4f}")
print(f"V^s(z_bar, theta) = {numeric_V_z.evalf():.4f}\n")


print("=== METHOD 2: Substituting first, then solving numerically ===")
# Substitute numbers directly into equations to eliminate parameters
eq1_numeric = eq1.subs(numeric_values)
eq2_numeric = eq2.subs(numeric_values)

# Solve the resulting linear system directly for the numerical answers
numeric_solutions = sp.solve([eq1_numeric, eq2_numeric], (V_0, V_z))

print(f"V_0 Answer = {float(numeric_solutions[V_0]):.4f}")
print(f"V_z Answer = {float(numeric_solutions[V_z]):.4f}")
