import matlab.engine

print("Connecting to MATLAB R2026a Engine...")
eng = matlab.engine.start_matlab()

# 1. Force MATLAB to look at your current working directory
# This allows it to find any local .m script files you write
eng.cd(r"C:\csharp_code_analysis", nargout=0)

print("\n--- Running Economic Equation Matrix Test ---")

# Define numeric values for our economics parameters
beta_val = 0.95
theta_val = 0.20
tau_val = 0.15
pz_val = 10.0

# Express the system as a matrix equation Ax = b
# Eq 1: (1 - beta*(1-theta))*V0 - beta*theta*Vz = -beta*theta*tau*pz
# Eq 2: -beta*V0 + Vz = pz
row1_V0 = 1.0 - beta_val * (1.0 - theta_val)
row1_Vz = -beta_val * theta_val
row2_V0 = -beta_val
row2_Vz = 1.0

b1 = -beta_val * theta_val * tau_val * pz_val
b2 = pz_val

# Convert Python lists into explicit MATLAB double arrays
A = matlab.double([[row1_V0, row1_Vz], [row2_V0, row2_Vz]])
b = matlab.double([[b1], [b2]])

# Use MATLAB's core mldivide matrix solver (equivalent to A \ b)
solution = eng.mldivide(A, b)

# Extract answers out of the returned MATLAB data structure
V0_ans = solution[0][0]
Vz_ans = solution[1][0]

print(f"MATLAB Solution for V0: {V0_ans:.4f}")
print(f"MATLAB Solution for Vz: {Vz_ans:.4f}")

# 2. Shut down the memory workspace link safely
eng.quit()
print("\nSession cleanly terminated.")
