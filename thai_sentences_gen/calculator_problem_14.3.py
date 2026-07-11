#!/usr/bin/env python3
"""
Python calculator for problem 14.13-style MOS equations.

Equations use:
gm = sqrt(mu_n Cox * (W/L) * I)

All currents are in amperes.
All voltages are in volts.
All resistances are in ohms.

page_010 - raz solutions


"""

from math import sqrt


# ------------------------------------------------------------
# Input parameters
# ------------------------------------------------------------

I_T = 1e-3          # total tail/reference current, 1 mA
W1 = 50.0
L1 = 0.5
WL_12 = W1 / L1    # (W/L) for M1, M2

# Example process parameter.
# Replace this with the value from the textbook/problem.
mu_n_Cox = 200e-6  # A/V^2

# Part (d) values
VDD_or_bias = 1.5
source_offset = 0.5
VTH_56 = 0.5


# ------------------------------------------------------------
# Basic MOS helper functions
# ------------------------------------------------------------

def gm(mu_c_ox: float, wl: float, current: float) -> float:
    """
    Compute MOS transconductance using:

        gm = sqrt(mu_n Cox * (W/L) * I)

    This follows the form used in the provided solution.
    """
    return sqrt(mu_c_ox * wl * current)


def parallel(a: float, b: float) -> float:
    """Parallel resistance."""
    return 1.0 / ((1.0 / a) + (1.0 / b))


# ------------------------------------------------------------
# Part (a)
# ------------------------------------------------------------

def compute_R12(mu_c_ox: float, wl_12: float, I_T: float) -> float:
    """
    For a three-stage ring oscillator, minimum low-frequency gain per stage is 2:

        gm_12 * R_12 = 2

    Therefore:

        R_12 = 2 / gm_12
    """
    gm_12 = gm(mu_c_ox, wl_12, I_T)
    R12 = 2.0 / gm_12
    return R12


# ------------------------------------------------------------
# Part (b)
# ------------------------------------------------------------

def compute_WL34(wl_12: float) -> float:
    """
    From the derivation:

        (W/L)_34 = 0.25^2 * (W/L)_12
    """
    return (0.25 ** 2) * wl_12


# ------------------------------------------------------------
# Part (c)
# ------------------------------------------------------------

def voltage_gain_with_negative_resistance(
    mu_c_ox: float,
    wl_12: float,
    wl_34: float,
    I_H: float,
    I_T: float,
    R12: float,
) -> float:
    """
    Gain expression:

        |Av| = gm_12 * (R12 || 1/gm_34)

    Here:
        gm_12 depends on I_H / 2
        gm_34 depends on I_T / 2
    """
    gm_12 = sqrt(2.0 * (I_H / 2.0) * mu_c_ox * wl_12)
    gm_34 = sqrt(2.0 * (I_T / 2.0) * mu_c_ox * wl_34)

    r_gm34 = 1.0 / gm_34
    effective_R = parallel(R12, r_gm34)

    return gm_12 * effective_R


def solve_IH_for_gain(
    mu_c_ox: float,
    wl_12: float,
    wl_34: float,
    I_T: float,
    R12: float,
    target_gain: float = 2.0,
) -> float:
    """
    Solve for I_H from:

        target_gain = gm_12 * (R12 || 1/gm_34)

    where:

        gm_12 = sqrt(mu_n Cox * (W/L)_12 * I_H)
    """

    gm_34 = sqrt(mu_c_ox * wl_34 * I_T)

    if gm_34 * R12 >= 1.0:
        raise ValueError(
            "Latch-up risk: gm_34 * R12 >= 1. "
            "The equation assumes gm_34 * R12 < 1."
        )

    effective_R = parallel(R12, 1.0 / gm_34)

    required_gm12 = target_gain / effective_R

    I_H = (required_gm12 ** 2) / (mu_c_ox * wl_12)

    return I_H


# ------------------------------------------------------------
# Part (d)
# ------------------------------------------------------------

def compute_WL56(
    I_T: float,
    mu_c_ox: float,
    VGS56: float,
    VTH56: float,
) -> float:
    """
    From:

        I_T / 2 = 1/2 * mu_n Cox * (W/L)_56 * (VGS56 - VTH56)^2

    Therefore:

        (W/L)_56 = I_T / [mu_n Cox * (VGS56 - VTH56)^2]
    """

    Vov = VGS56 - VTH56

    if Vov <= 0:
        raise ValueError("VGS56 must be greater than VTH56.")

    return I_T / (mu_c_ox * Vov ** 2)


# ------------------------------------------------------------
# Main calculation
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("Problem 14.13 MOS Calculator")
    print("=" * 70)

    print(f"I_T       = {I_T:.4e} A")
    print(f"(W/L)_12  = {WL_12:.4f}")
    print(f"mu_n Cox  = {mu_n_Cox:.4e} A/V^2")
    print()

    # Part (a)
    gm12 = gm(mu_n_Cox, WL_12, I_T)
    R12 = compute_R12(mu_n_Cox, WL_12, I_T)

    print("Part (a)")
    print(f"gm_12 = {gm12:.6e} S")
    print(f"R_12  = {R12:.6f} ohms")
    print()

    # Part (b)
    WL_34 = compute_WL34(WL_12)

    print("Part (b)")
    print(f"(W/L)_34 = {WL_34:.6f}")
    print()

    # Part (c)
    I_H = solve_IH_for_gain(
        mu_c_ox=mu_n_Cox,
        wl_12=WL_12,
        wl_34=WL_34,
        I_T=I_T,
        R12=R12,
        target_gain=2.0,
    )

    Av_check = voltage_gain_with_negative_resistance(
        mu_c_ox=mu_n_Cox,
        wl_12=WL_12,
        wl_34=WL_34,
        I_H=I_H,
        I_T=I_T,
        R12=R12,
    )

    print("Part (c)")
    print(f"I_H required = {I_H:.6e} A")
    print(f"I_H required = {I_H * 1e6:.6f} uA")
    print(f"Gain check   = {Av_check:.6f}")
    print()

    # Part (d)
    VGS56 = VDD_or_bias - source_offset

    WL_56 = compute_WL56(
        I_T=I_T,
        mu_c_ox=mu_n_Cox,
        VGS56=VGS56,
        VTH56=VTH_56,
    )

    print("Part (d)")
    print(f"VGS_56    = {VGS56:.6f} V")
    print(f"VTH_56    = {VTH_56:.6f} V")
    print(f"Vov_56    = {VGS56 - VTH_56:.6f} V")
    print(f"(W/L)_56  = {WL_56:.6f}")
    print()


if __name__ == "__main__":
    main()