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

    chg = None
    wall = None

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
                "d": None
            })
            continue

        trend_changed = False

        # --- Detect transitions ---
        if bar.idx > last_idx:
            if state != "up":
                state = "up"
                chg = bar.seq
                trend_changed = True
                up_high_price = bar.price
                up_high_seq = bar.seq

        elif bar.idx < last_idx:
            if state != "down":
                state = "down"
                chg = bar.seq
                trend_changed = True
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

        # --- Determine extreme for this bar ---
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
            "d": d
        })

    return results


if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    for r in results:
        print(
            f"{r['seq']}  {r['date']}  {r['price']}  {r['idx']}  "
            f"state={r['state']}  chg={r['chg']}  "
            f"lph={r['lph_price']} at seq {r['lph_seq']}  "
            f"ldl={r['ldl_price']} at seq {r['ldl_seq']}  "
            f"dsex={r['dsex']}  wall={r['wall']}  d={r['d']}"
        )
