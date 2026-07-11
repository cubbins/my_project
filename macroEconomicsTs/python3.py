
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
# 2. Numeric parameter values (MUST come before steady-state)
# ============================================================

a_ = 1.0
eta_ = 0.3
g_ = 0.2
tau_ = 0.1
beta_ = 0.05
gamma_ = 0.5

# ============================================================
# 3. Substitute parameters into symbolic expressions
# ============================================================

dot_z_num = dot_z.subs({a: a_, eta: eta_, g: g_})
dot_c_num = dot_c.subs({
    a: a_, eta: eta_, g: g_,
    tau: tau_, beta: beta_, gamma: gamma_
})

# ============================================================
# 4. Solve steady state correctly
# ============================================================

# Solve dot(z)=0 for c(z)
c_from_z = sp.solve(sp.Eq(dot_z_num, 0), c)[0]

# Substitute c(z) into dot(c)=0
eq_z_only = sp.simplify(dot_c_num.subs(c, c_from_z))

# Solve for z_bar
z_bar_val = float(sp.nsolve(eq_z_only, 1.0))  # initial guess 1.0

# Compute c_bar
c_bar_val = float(c_from_z.subs(z, z_bar_val))

print("\nSteady state:")
print("z_bar =", z_bar_val)
print("c_bar =", c_bar_val)


# ============================================================
# 3. Jacobian and eigenstructure at steady state
# ============================================================

F1 = dot_z
F2 = dot_c
J = sp.Matrix([
    [sp.diff(F1, z), sp.diff(F1, c)],
    [sp.diff(F2, z), sp.diff(F2, c)]
])

# Parameter values
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

# Substitute parameters and steady state into Jacobian
J_num = J.subs({
    a: a_,
    eta: eta_,
    g: g_,
    tau: tau_,
    beta: beta_,
    gamma: gamma_,
    z: z_bar_val,
    c: c_bar_val
}).evalf()

print("\nJacobian at steady state:")
sp.pprint(J_num)

# Eigenvalues and eigenvectors
eigs = J_num.eigenvects()
print("\nEigenvalues and eigenvectors:")
for val, mult, vecs in eigs:
    print("λ =", val)
    for v in vecs:
        print("  v =", v)

# ============================================================
# 4. Numerical dynamics with positivity enforcement
# ============================================================

def dynamics(t, state):
    z_val, c_val = state

    if z_val <= 1e-8:
        z_val = 1e-8
    if c_val <= 1e-8:
        c_val = 1e-8

    phi_g_val = g_ * a_ * z_val**(eta_ - 1)
    phi_K_val = (1 - g_) * a_ * z_val**(eta_ - 1) - c_val
    dot_z_val = phi_g_val - phi_K_val

    phi_C_val = ((1 - tau_) * a_ * (1 - eta_) * z_val**(eta_ - 1) - beta_) / (1 - gamma_)
    dot_c_val = c_val * (phi_C_val - phi_K_val)

    return [dot_z_val, dot_c_val]

# ============================================================
# 5. Phase diagram: nullclines and vector field
# ============================================================

# Grid for phase diagram
z_min, z_max = 0.1, 3.0
c_min, c_max = 0.01, 0.5
Nz, Nc = 25, 25

Z = np.linspace(z_min, z_max, Nz)
C = np.linspace(c_min, c_max, Nc)
ZZ, CC = np.meshgrid(Z, C)

dZ = np.zeros_like(ZZ)
dC = np.zeros_like(CC)

for i in range(Nc):
    for j in range(Nz):
        dz_val, dc_val = dynamics(0, [ZZ[i, j], CC[i, j]])
        dZ[i, j] = dz_val
        dC[i, j] = dc_val

# Nullclines: dot(z)=0 and dot(c)=0
# Solve for c as function of z for dot(z)=0: dot_z = 0 => c = ...
c_null_z = sp.solve(sp.Eq(dot_z.subs({
    a: a_,
    eta: eta_,
    g: g_
}), 0), c)
c_null_z_func = sp.lambdify(z, c_null_z[0], 'numpy')

# For dot(c)=0: phi_C = phi_K
c_null_c = sp.solve(sp.Eq(phi_C.subs({
    a: a_,
    eta: eta_,
    g: g_,
    tau: tau_,
    beta: beta_,
    gamma: gamma_
}), phi_K.subs({
    a: a_,
    eta: eta_,
    g: g_
})), c)
c_null_c_func = sp.lambdify(z, c_null_c[0], 'numpy')

# ============================================================
# 6. Simulated trajectory near steady state
# ============================================================

z0 = z_bar_val * 0.8
c0 = c_bar_val * 1.2

sol = solve_ivp(
    dynamics,
    (0, 200),
    [z0, c0],
    method='LSODA',
    dense_output=True,
    max_step=0.1
)

t = np.linspace(0, 200, 1000)
z_path, c_path = sol.sol(t)

# ============================================================
# 7. Plot: phase diagram in (z, c)
# ============================================================

plt.figure(figsize=(10, 5))

# Vector field
speed = np.sqrt(dZ**2 + dC**2)
plt.streamplot(ZZ, CC, dZ, dC, color='lightgray', density=1.0, arrowsize=1)

# Nullclines
z_line = np.linspace(z_min, z_max, 400)
plt.plot(z_line, c_null_z_func(z_line), 'r-', label='dot(z)=0')
plt.plot(z_line, c_null_c_func(z_line), 'b-', label='dot(c)=0')

# Steady state
plt.plot(z_bar_val, c_bar_val, 'ko', label='steady state')

# Trajectory
plt.plot(z_path, c_path, 'g-', label='trajectory')

plt.xlim(z_min, z_max)
plt.ylim(c_min, c_max)
plt.xlabel('z = K_g / K')
plt.ylabel('c = C / K')
plt.title('Phase Diagram in (z, c)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 8. Optional: growth-rate phase diagram (z, dot(z)) and (z, dot(c))
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(z_line, [dynamics(0, [zz, c_bar_val])[0] for zz in z_line], label='dot(z) at c=c_bar')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(z_bar_val, color='k', linewidth=0.5, linestyle='--')
plt.xlabel('z')
plt.ylabel('dot(z)')
plt.title('Growth of z around steady state')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(z_line, [dynamics(0, [zz, c_bar_val])[1] for zz in z_line], label='dot(c) at c=c_bar')
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(z_bar_val, color='k', linewidth=0.5, linestyle='--')
plt.xlabel('z')
plt.ylabel('dot(c)')
plt.title('Growth of c around steady state')
plt.grid(True)
plt.tight_layout()
plt.show()


