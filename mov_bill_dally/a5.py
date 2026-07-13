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

    # Track highs during UP segments
    up_high_price = None
    up_high_seq = None

    # Track lows during DOWN segments
    down_low_price = None
    down_low_seq = None

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
                "last_up_high_seq": None,
                "last_down_low_price": None,
                "last_down_low_seq": None,
                "days_since_extreme": None
            })
            continue

        # --- Detect transitions ---
        if bar.idx > last_idx:
            if state != "up":
                state = "up"
                changed_at_seq = bar.seq
                # Reset UP high tracking
                up_high_price = bar.price
                up_high_seq = bar.seq

        elif bar.idx < last_idx:
            if state != "down":
                state = "down"
                changed_at_seq = bar.seq
                # Reset DOWN low tracking
                down_low_price = bar.price
                down_low_seq = bar.seq

        # --- Update segment highs/lows ---
        if state == "up":
            if up_high_price is None or bar.price > up_high_price:
                up_high_price = bar.price
                up_high_seq = bar.seq

        elif state == "down":
            if down_low_price is None or bar.price < down_low_price:
                down_low_price = bar.price
                down_low_seq = bar.seq

        last_idx = bar.idx

        # --- Determine what to report ---
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
            days_since_extreme = None
        else:
            days_since_extreme = bar.seq - extreme_seq

        results.append({
            "seq": bar.seq,
            "date": bar.date,
            "price": bar.price,
            "idx": bar.idx,
            "state": state,
            "changed_at_seq": changed_at_seq,
            "last_up_high_price": up_high_price if state == "down" else None,
            "last_up_high_seq": up_high_seq if state == "down" else None,
            "last_down_low_price": down_low_price if state == "up" else None,
            "last_down_low_seq": down_low_seq if state == "up" else None,
            "days_since_extreme": days_since_extreme
        })

    return results


if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    for r in results:
        if r["state"] == "down":
            if r["last_up_high_price"] is None:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_up_high=None  days_since_extreme={r['days_since_extreme']}")
            else:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_up_high={r['last_up_high_price']} at seq {r['last_up_high_seq']}  "
                      f"days_since_extreme={r['days_since_extreme']}")
        else:
            if r["last_down_low_price"] is None:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_down_low=None  days_since_extreme={r['days_since_extreme']}")
            else:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_down_low={r['last_down_low_price']} at seq {r['last_down_low_seq']}  "
                      f"days_since_extreme={r['days_since_extreme']}")
