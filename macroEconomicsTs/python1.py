
# git config --global user.name "Your Name"
# git config --global user.email "you@example.com"
# git status
# Add a specific file
# git add filename.txt
# Add a specific folder
# git add src/
# Add everything in the current directory
# git add .
# git commit -m "Your descriptive message here"
# git push origin main

# git log --oneline See commit history
# git pull origin mainPull latest changes from GitHub first
# git diffSee unstaged changes
# git branchSee which branch you're on


# git remote add origin https://github.com/yourusername/your-repo.git
# git branch -M main
# git push -u origin main





import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# 1. Symbolic model setup
# =========================

# Parameters (symbolic)
a, eta, g, tau, beta, gamma = sp.symbols('a eta g tau beta gamma', positive=True)
z, c = sp.symbols('z c', positive=True)

# Production per unit of private capital: Y/K
Y_over_K = a * z**(1 - eta)

# Growth of public capital: phi_g = K_g_dot / K_g
phi_g = g * a * z**(eta - 1)

# Growth of private capital: phi_K = K_dot / K
phi_K = (1 - g) * a * z**(eta - 1) - c

# z dynamics: z_dot = z*(phi_g - phi_K)
dot_z = sp.simplify(phi_g - phi_K)

# Euler equation: C_dot / C
phi_C = ((1 - tau) * a * (1 - eta) * z**(eta - 1) - beta) / (1 - gamma)

# c dynamics: (c_dot / c) = phi_C - phi_K
dot_c_over_c = sp.simplify(phi_C - phi_K)
dot_c = sp.simplify(dot_c_over_c * c)

print("Symbolic dynamics:")
print("dot(z) =", dot_z)
print("dot(c) =", dot_c)

# Jacobian of the system [dot_z, dot_c] w.r.t. [z, c]
F1 = dot_z
F2 = dot_c
J = sp.Matrix([
    [sp.diff(F1, z), sp.diff(F1, c)],
    [sp.diff(F2, z), sp.diff(F2, c)]
])

print("\nJacobian matrix J(z, c):")
sp.pprint(J)

# =========================
# 2. Numerical parameterization
# =========================

# Choose numerical values for parameters
params = {
    'a': 1.0,
    'eta': 0.3,
    'g': 0.2,
    'tau': 0.1,
    'beta': 0.05,
    'gamma': 0.5
}

a_val = params['a']
eta_val = params['eta']
g_val = params['g']
tau_val = params['tau']
beta_val = params['beta']
gamma_val = params['gamma']

# =========================
# 3. Numerical dynamics
# =========================

def dynamics(t, state):
    z_val, c_val = state
    a_ = a_val
    eta_ = eta_val
    g_ = g_val
    tau_ = tau_val
    beta_ = beta_val
    gamma_ = gamma_val

    # Growth rates
    phi_g_val = g_ * a_ * z_val**(eta_ - 1)
    phi_K_val = (1 - g_) * a_ * z_val**(eta_ - 1) - c_val
    dot_z_val = phi_g_val - phi_K_val

    phi_C_val = ((1 - tau_) * a_ * (1 - eta_) * z_val**(eta_ - 1) - beta_) / (1 - gamma_)
    dot_c_val = c_val * (phi_C_val - phi_K_val)

    return [dot_z_val, dot_c_val]

# =========================
# 4. Simulation
# =========================

# Initial conditions
z0 = 0.5
c0 = 0.1
y0 = [z0, c0]

# Time span
t_span = (0, 200)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval, dense_output=True)

z_path = sol.y[0]
c_path = sol.y[1]

# =========================
# 5. Plot results
# =========================

plt.figure(figsize=(10, 5))
plt.plot(t_eval, z_path, label='z(t) = K_g / K')
plt.plot(t_eval, c_path, label='c(t) = C / K')
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Time')
plt.ylabel('Ratios')
plt.title('Transitional Dynamics of z and c')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
