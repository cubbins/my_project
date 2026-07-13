import pandas as pd
import matplotlib.pyplot as plt

# Input CSV from LIM CUDA run
csv_path = "lim_waveforms.csv"

df = pd.read_csv(csv_path)

# Basic cleanup: avoid issues if there are exact zeros on log scale
eps = 1e-30
for col in ["V_probe0", "V_probe1", "V_probe2"]:
    df[col + "_abs"] = df[col].abs().clip(lower=eps)

# ------------------------------------------------------------
# Plot 1: Linear-scale voltage vs time
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(df["time_s"], df["V_probe0"], label="probe0")
plt.plot(df["time_s"], df["V_probe1"], label="probe1")
plt.plot(df["time_s"], df["V_probe2"], label="probe2")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("LIM CUDA Probe Voltages vs Time (Linear Scale)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lim_probes_linear.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Plot 2: Semilog plot to show tiny early arrivals
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.semilogy(df["time_s"], df["V_probe0_abs"], label="|probe0|")
plt.semilogy(df["time_s"], df["V_probe1_abs"], label="|probe1|")
plt.semilogy(df["time_s"], df["V_probe2_abs"], label="|probe2|")
plt.xlabel("Time (s)")
plt.ylabel("Absolute Voltage (V)")
plt.title("LIM CUDA Probe Voltages vs Time (Semilog Scale)")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("lim_probes_semilog.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Plot 3: Normalized responses, relative to final probe0
# Helps compare propagation shape independent of magnitude
# ------------------------------------------------------------
ref = abs(df["V_probe0"].iloc[-1])
if ref < eps:
    ref = 1.0

plt.figure(figsize=(10, 6))
plt.plot(df["time_s"], df["V_probe0"] / ref, label="probe0 / final_probe0")
plt.plot(df["time_s"], df["V_probe1"] / ref, label="probe1 / final_probe0")
plt.plot(df["time_s"], df["V_probe2"] / ref, label="probe2 / final_probe0")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Voltage")
plt.title("Normalized LIM CUDA Probe Responses")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lim_probes_normalized.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Optional: report final values
# ------------------------------------------------------------
print("Final values:")
print(df.iloc[-1][["step", "time_s", "V_probe0", "V_probe1", "V_probe2"]])
