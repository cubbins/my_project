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
            # seq, date, price, index
            seq = int(parts[0])
            date = parts[1]
            price = float(parts[2])
            idx = int(parts[3])
            bars.append(Bar(seq, date, price, idx))
    return bars

def analyze_trend(bars):
    state = "unknown"      # "up" or "down"
    last_idx = None

    # track current up-trend segment
    up_start = None        # index in bars where current up-trend starts
    up_high_price = None
    up_high_date = None

    results = []  # list of dicts with per-bar info

    for i, bar in enumerate(bars):
        if last_idx is None:
            # first bar: no trend yet
            last_idx = bar.idx
            results.append({
                "seq": bar.seq,
                "date": bar.date,
                "price": bar.price,
                "idx": bar.idx,
                "state": state,
                "last_up_high_price": None,
                "last_up_high_date": None,
            })
            continue

        if bar.idx > last_idx:
            # index increasing
            if state != "up":
                # new up-trend starts at previous bar
                state = "up"
                up_start = i - 1
                up_high_price = bars[up_start].price
                up_high_date = bars[up_start].date

            # update high within current up-trend
            if bar.price > up_high_price:
                up_high_price = bar.price
                up_high_date = bar.date

        elif bar.idx < last_idx:
            # index decreasing
            if state == "up":
                # trend flips from up to down
                state = "down"
                # up_high_* already holds the max price since up_start

        # if bar.idx == last_idx, trend doesn't change; we just keep state

        last_idx = bar.idx

        # when state is down, we look back only to the last up-trend
        if state == "down":
            last_up_high_price = up_high_price
            last_up_high_date = up_high_date
        else:
            last_up_high_price = None
            last_up_high_date = None

        results.append({
            "seq": bar.seq,
            "date": bar.date,
            "price": bar.price,
            "idx": bar.idx,
            "state": state,
            "last_up_high_price": last_up_high_price,
            "last_up_high_date": last_up_high_date,
        })

    return results

if __name__ == "__main__":
    bars = parse_file("temp.txt")
    results = analyze_trend(bars)

    # Example: print only rows where state is down
    for r in results:
        if r["state"] == "down":
            print(
                f"{r['seq']:3d} {r['date']} price={r['price']:8.3f} "
                f"idx={r['idx']:3d} state={r['state']:4s} "
                f"last_up_high={r['last_up_high_price']} on {r['last_up_high_date']}"
            )
