from dataclasses import dataclass

@dataclass
class Bar:
    seq: int
    date: str
    price: float
    idx: int

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

def analyze_trend(bars):
    state = "unknown"
    last_idx = None

    # Track the most recent up‑trend segment
    up_start = None
    up_high_price = None
    up_high_seq = None

    results = []

    for i, bar in enumerate(bars):

        if last_idx is None:
            last_idx = bar.idx
            results.append({
                "seq": bar.seq,
                "state": state,
                "announce": None
            })
            continue

        announce = None

        # --- Detect UP movement ---
        if bar.idx > last_idx:
            if state != "up":
                # New up‑trend begins at previous bar
                state = "up"
                up_start = i - 1
                up_high_price = bars[up_start].price
                up_high_seq = bars[up_start].seq

            # Update high inside up‑trend
            if bar.price > up_high_price:
                up_high_price = bar.price
                up_high_seq = bar.seq

        # --- Detect DOWN movement ---
        elif bar.idx < last_idx:
            if state == "up":
                # Trend flips UP → DOWN
                state = "down"
                announce = (
                    f"Trend change at seq {bar.seq}: "
                    f"last up-trend high was {up_high_price} at seq {up_high_seq}"
                )

        # idx == last_idx → no change in trend

        last_idx = bar.idx

        results.append({
            "seq": bar.seq,
            "state": state,
            "announce": announce
        })

    return results


if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    # Print only the announcements
    for r in results:
        if r["announce"]:
            print(r["announce"])
