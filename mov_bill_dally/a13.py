from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# DATA STRUCTURES
# ------------------------------------------------------------

@dataclass
class Bar:
    seq: int
    date: str
    price: float
    idx: int


# ------------------------------------------------------------
# FILE PARSER
# ------------------------------------------------------------

def parse_file(path: str):
    bars = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            seq = int(parts[0])
            date = parts[1]
            price = float(parts[2])
            idx = int(parts[3])
            bars.append(Bar(seq, date, price, idx))
    return bars


# ------------------------------------------------------------
# TREND ANALYSIS ENGINE
# ------------------------------------------------------------

def analyze_trend(bars):
    state = "unknown"
    last_idx = None

    up_high_price = None
    up_high_seq = None

    down_low_price = None
    down_low_seq = None

    chg = None
    wall = None

    switch_price = None

    results = []

    for i, bar in enumerate(bars):

        if last_idx is None:
            last_idx = bar.idx
            results.append({
                "seq": bar.seq,
                "date": bar.date,
                "price": bar.price,
                "idx": bar.idx,
                "state": state,
                "chg": None,
                "lph_price": None,
                "lph_seq": None,
                "ldl_price": None,
                "ldl_seq": None,
                "dsex": None,
                "wall": None,
                "d": None,
                "switch_price": None,
                "switch_diff": None
            })
            continue

        trend_changed = False
        prior_state = state

        # --- Detect transitions ---
        if bar.idx > last_idx:
            if state != "up":
                state = "up"
                chg = bar.seq
                trend_changed = True

        elif bar.idx < last_idx:
            if state != "down":
                state = "down"
                chg = bar.seq
                trend_changed = True

        # --- Update highs/lows BEFORE resetting ---
        if prior_state == "up":
            if up_high_price is None or bar.price > up_high_price:
                up_high_price = bar.price
                up_high_seq = bar.seq

        elif prior_state == "down":
            if down_low_price is None or bar.price < down_low_price:
                down_low_price = bar.price
                down_low_seq = bar.seq

        # --- Handle transition resets ---
        if trend_changed:

            switch_price = bar.price  # NEW

            if state == "up":
                up_high_price = bar.price
                up_high_seq = bar.seq

            elif state == "down":
                down_low_price = bar.price
                down_low_seq = bar.seq

        last_idx = bar.idx

        # --- Determine extreme ---
        if state == "down":
            extreme_price = up_high_price
            extreme_seq = up_high_seq
        elif state == "up":
            extreme_price = down_low_price
            extreme_seq = down_low_seq
        else:
            extreme_price = None
            extreme_seq = None

        if extreme_seq is None:
            dsex = None
        else:
            dsex = bar.seq - extreme_seq

        # --- WALL logic ---
        if trend_changed:
            wall = dsex
            k = 0
        else:
            if chg is None:
                k = None
            else:
                k = bar.seq - chg

        if wall is None or k is None:
            d = None
        else:
            d = wall - k

        # --- switch_diff logic ---
        if switch_price is None:
            switch_diff = None
        else:
            raw_diff = bar.price - switch_price
            switch_diff = abs(raw_diff) if state == "down" else raw_diff

        results.append({
            "seq": bar.seq,
            "date": bar.date,
            "price": bar.price,
            "idx": bar.idx,
            "state": state,
            "chg": chg,
            "lph_price": up_high_price if state == "down" else None,
            "lph_seq": up_high_seq if state == "down" else None,
            "ldl_price": down_low_price if state == "up" else None,
            "ldl_seq": down_low_seq if state == "up" else None,
            "dsex": dsex,
            "wall": wall,
            "d": d,
            "switch_price": switch_price,
            "switch_diff": switch_diff
        })

    return results


# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

def plot_results(results):
    seq = [r["seq"] for r in results]
    price = [r["price"] for r in results]
    wall = [r["wall"] for r in results]
    d = [r["d"] for r in results]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(seq, price, color="blue", linewidth=2, label="Price")
    ax1.set_xlabel("Sequence Number")
    ax1.set_ylabel("Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(seq, wall, color="green", linestyle="--", linewidth=2, label="Wall")
    ax2.plot(seq, d, color="red", linestyle="-.", linewidth=2, label="d")
    ax2.set_ylabel("Wall / d", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    plt.title("Price, Wall, and d Over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# CORRELATION CALCULATIONS
# ------------------------------------------------------------

def compute_correlations(results):
    price = np.array([r["price"] for r in results], dtype=float)
    wall = np.array([r["wall"] for r in results], dtype=float)
    d = np.array([r["d"] for r in results], dtype=float)

    mask = (~np.isnan(price)) & (~np.isnan(wall)) & (~np.isnan(d))
    price = price[mask]
    wall = wall[mask]
    d = d[mask]

    corr_pw = np.corrcoef(price, wall)[0, 1]
    corr_wd = np.corrcoef(wall, d)[0, 1]
    corr_pd = np.corrcoef(price, d)[0, 1]

    print("\n=== CORRELATIONS ===")
    print(f"Correlation(price, wall) = {corr_pw:.6f}")
    print(f"Correlation(wall, d)    = {corr_wd:.6f}")
    print(f"Correlation(price, d)   = {corr_pd:.6f}")

    return corr_pw, corr_wd, corr_pd


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    # --------------------------------------------------------
    # Build 2D table WITH BLANK ROWS ON STATE TRANSITIONS
    # --------------------------------------------------------
    columns = list(results[0].keys())
    table = []

    last_state = None

    for r in results:

        if last_state is not None and r["state"] != last_state:
            table.append([None] * len(columns))

        table.append([r[col] for col in columns])

        last_state = r["state"]

    # --------------------------------------------------------
    # FIRST REPORT (already working)
    # --------------------------------------------------------
    last_state = None

    for r in results:

        if last_state is not None and r["state"] != last_state:
            print()

        print(
            f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
            f"state={r['state']}  chg={r['chg']}  "
            f"lph={r['lph_price']} at seq {r['lph_seq']}  "
            f"ldl={r['ldl_price']} at seq {r['ldl_seq']}  "
            f"dsex={r['dsex']}  wall={r['wall']}  d={r['d']}"
        )

        if r['switch_diff'] is None:
            sd = None
        else:
            sd = f"{r['switch_diff']:.3f}"

        print(f"    switch_price={r['switch_price']}  switch_diff={sd}")

        last_state = r["state"]

    # --------------------------------------------------------
    # SECOND REPORT (FIXED FORMATTING)
    # --------------------------------------------------------
    print("\n=== SUMMARY TABLE (seq, date, price, idx, state, wall, d, switch_price, switch_diff) ===")

    last_state = None

    for r in results:

        if last_state is not None and r["state"] != last_state:
            print()

        # Safe formatting for wall and d
        wall_val = f"{r['wall']:.3f}" if isinstance(r['wall'], (int, float)) else "None"
        d_val    = f"{r['d']:.3f}"    if isinstance(r['d'],    (int, float)) else "None"

        # Safe formatting for switch_diff
        if r['switch_diff'] is None:
            sd = "None"
        else:
            sd = f"{r['switch_diff']:.3f}"

        print(
            f"{r['seq']:5d}  {r['date']}  {r['price']:10.4f}  {r['idx']:3d}  "
            f"{r['state']:6s}  {wall_val:>8}  {d_val:>8}  "
            f"{r['switch_price']}  {sd}"
        )

        last_state = r["state"]

    compute_correlations(results)
    #plot_results(results)
