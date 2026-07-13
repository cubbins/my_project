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

    changed_at_seq = None

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
                "changed_at_seq": None,
                "last_up_high_price": None,
                "last_up_high_seq": None
            })
            continue

        # --- Detect UP movement ---
        if bar.idx > last_idx:
            if state != "up":
                # New up‑trend begins at previous bar
                state = "up"
                up_start = i - 1
                up_high_price = bars[up_start].price
                up_high_seq = bars[up_start].seq
                changed_at_seq = bar.seq

            # Update high inside up‑trend
            if bar.price > up_high_price:
                up_high_price = bar.price
                up_high_seq = bar.seq

        # --- Detect DOWN movement ---
        elif bar.idx < last_idx:
            if state == "up":
                # Trend flips UP → DOWN
                state = "down"
                changed_at_seq = bar.seq

        # idx == last_idx → no change in trend

        last_idx = bar.idx

        # When down, we report the last up‑trend high
        if state == "down":
            lup = up_high_price
            lup_seq = up_high_seq
        else:
            lup = None
            lup_seq = None

        results.append({
            "seq": bar.seq,
            "date": bar.date,
            "price": bar.price,
            "idx": bar.idx,
            "state": state,
            "changed_at_seq": changed_at_seq,
            "last_up_high_price": lup,
            "last_up_high_seq": lup_seq
        })

    return results


if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    for r in results:
        if r["last_up_high_price"] is None:
            print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                  f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                  f"last_up_high=None")
        else:
            print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                  f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                  f"last_up_high={r['last_up_high_price']} at seq {r['last_up_high_seq']}")
