

import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# 1. Symbolic model setup
# ============================================================

a, eta, g, tau, beta, gamma = sp.symbols('a eta g tau beta gamma', positive=True)
z, c = sp.symbols('z c', positive=True)

phi_g = g * a * z**(eta - 1)
phi_K = (1 - g) * a * z**(eta - 1) - c
dot_z = sp.simplify(phi_g - phi_K)

phi_C = ((1 - tau) * a * (1 - eta) * z**(eta - 1) - beta) / (1 - gamma)
dot_c = sp.simplify(c * (phi_C - phi_K))

print("dot(z) =", dot_z)
print("dot(c) =", dot_c)

# ============================================================
# 2. Numerical parameters
# ============================================================

params = dict(
    a=1.0,
    eta=0.3,
    g=0.2,
    tau=0.1,
    beta=0.05,
    gamma=0.5
)

a_ = params['a']
eta_ = params['eta']
g_ = params['g']
tau_ = params['tau']
beta_ = params['beta']
gamma_ = params['gamma']

# ============================================================
# 3. Numerical dynamics with positivity enforcement
# ============================================================

def dynamics(t, state):
    z, c = state

    # Enforce positivity floor
    if z <= 1e-8:
        z = 1e-8
    if c <= 1e-8:
        c = 1e-8

    phi_g_val = g_ * a_ * z**(eta_ - 1)
    phi_K_val = (1 - g_) * a_ * z**(eta_ - 1) - c
    dot_z_val = phi_g_val - phi_K_val

    phi_C_val = ((1 - tau_) * a_ * (1 - eta_) * z**(eta_ - 1) - beta_) / (1 - gamma_)
    dot_c_val = c * (phi_C_val - phi_K_val)

    return [dot_z_val, dot_c_val]

# Stop integration if z hits zero
def hit_zero_z(t, state):
    return state[0] - 1e-8
hit_zero_z.terminal = True
hit_zero_z.direction = -1

# ============================================================
# 4. Simulation
# ============================================================

z0 = 0.5
c0 = 0.1

sol = solve_ivp(
    dynamics,
    (0, 200),
    [z0, c0],
    method='LSODA',
    events=[hit_zero_z],
    dense_output=True,
    max_step=0.1
)

t = np.linspace(0, sol.t_events[0][0] if sol.t_events[0].size > 0 else 200, 1000)
z_path, c_path = sol.sol(t)

# ============================================================
# 5. Plot
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(t, z_path, label='z(t) = K_g / K')
plt.plot(t, c_path, label='c(t) = C / K')
plt.xlabel("Time")
plt.ylabel("Ratios")
plt.title("Transitional Dynamics with Positivity Enforcement")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
