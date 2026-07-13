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
                "last_down_low_seq": None
            })
            continue

        # --- Detect transitions ---
        if bar.idx > last_idx:
            # Index rising → possible DOWN → UP transition
            if state != "up":
                state = "up"
                changed_at_seq = bar.seq
                # Reset UP high tracking
                up_high_price = bar.price
                up_high_seq = bar.seq
        elif bar.idx < last_idx:
            # Index falling → possible UP → DOWN transition
            if state != "down":
                state = "down"
                changed_at_seq = bar.seq
                # Reset DOWN low tracking
                down_low_price = bar.price
                down_low_seq = bar.seq

        # --- Update segment highs/lows ---
        if state == "up":
            # Update UP high
            if up_high_price is None or bar.price > up_high_price:
                up_high_price = bar.price
                up_high_seq = bar.seq
        elif state == "down":
            # Update DOWN low
            if down_low_price is None or bar.price < down_low_price:
                down_low_price = bar.price
                down_low_seq = bar.seq

        last_idx = bar.idx

        # --- Determine what to report ---
        if state == "down":
            lup = up_high_price
            lup_seq = up_high_seq
            ldl = None
            ldl_seq = None
        elif state == "up":
            lup = None
            lup_seq = None
            ldl = down_low_price
            ldl_seq = down_low_seq
        else:
            lup = lup_seq = ldl = ldl_seq = None

        results.append({
            "seq": bar.seq,
            "date": bar.date,
            "price": bar.price,
            "idx": bar.idx,
            "state": state,
            "changed_at_seq": changed_at_seq,
            "last_up_high_price": lup,
            "last_up_high_seq": lup_seq,
            "last_down_low_price": ldl,
            "last_down_low_seq": ldl_seq
        })

    return results


if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    for r in results:
        if r["state"] == "down":
            # Report UP → DOWN info
            if r["last_up_high_price"] is None:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_up_high=None")
            else:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_up_high={r['last_up_high_price']} at seq {r['last_up_high_seq']}")
        else:
            # Report DOWN → UP info
            if r["last_down_low_price"] is None:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_down_low=None")
            else:
                print(f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
                      f"state={r['state']}  changed_at_seq={r['changed_at_seq']}  "
                      f"last_down_low={r['last_down_low_price']} at seq {r['last_down_low_seq']}")
