#!/usr/bin/env python3
"""
Analyze memory_thought_report.json.

Produces:
- memory_summary_report.md
- memory_detailed_report.md
- memory_statistics.json
- memory_items.csv
- call_statistics.csv

Run:
python analyze_memory_thought_report.py \
  --input ./dostoevsky_influenced_run/2026-07-07_12-04-42/memory_thought_report.json \
  --output_dir memory_thought_analysis

python analyze_memory_7_8_7_00.py \
  --input ./dostoevsky_influenced_run/2026-07-07_12-04-42/memory_thought_report.json \
  --output_dir memory_thought_analysis

"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime


MEMORY_HEADER_RE = re.compile(
    r"\[Memory item (?P<item>\d+); (?P<body>[^\]]+)\]",
    re.MULTILINE,
)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def safe_mean(values):
    return statistics.mean(values) if values else None


def safe_median(values):
    return statistics.median(values) if values else None


def numeric_summary(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return {
        "count": len(values),
        "sum": sum(values) if values else 0,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": safe_mean(values),
        "median": safe_median(values),
    }


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def parse_memory_items(user_prompt: str, current_call: dict) -> list[dict]:
    items = []

    for match in MEMORY_HEADER_RE.finditer(user_prompt or ""):
        item = {
            "current_call_type": current_call.get("call_type"),
            "current_call_timestamp": current_call.get("timestamp"),
            "current_run_id": current_call.get("run_id"),
            "memory_item_number": int(match.group("item")),
        }

        body = match.group("body")

        for part in body.split(";"):
            if "=" not in part:
                continue

            key, value = part.strip().split("=", 1)
            key = key.strip()
            value = value.strip()

            if key in {"thought_seconds", "weight"}:
                try:
                    value = float(value)
                except ValueError:
                    pass

            elif key in {"retained_chars", "original_output_chars"}:
                try:
                    value = int(value)
                except ValueError:
                    pass

            item[key] = value

        retained = item.get("retained_chars")
        original = item.get("original_output_chars")

        if isinstance(retained, int) and isinstance(original, int) and original > 0:
            item["retention_ratio"] = retained / original
            item["discarded_chars"] = original - retained
            item["discarded_ratio"] = 1.0 - item["retention_ratio"]

        items.append(item)

    return items


def extract_current_request(user_prompt: str) -> str:
    marker = "CURRENT REQUEST"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1].strip()
    return user_prompt.strip()


def has_memory_block(user_prompt: str) -> bool:
    return "PERSISTENT INTERMEDIATE MEMORY FROM PRIOR RUNS" in (user_prompt or "")


def analyze(data: dict) -> dict:
    calls = data.get("calls", [])

    call_rows = []
    memory_rows = []

    for call in calls:
        metadata = call.get("metadata", {})
        user_prompt = call.get("user_prompt", "")
        output = call.get("output", "")

        memory_items = parse_memory_items(user_prompt, call)
        memory_rows.extend(memory_items)

        prompt_chars = call.get("prompt_chars", 0)
        memory_chars = metadata.get("memory_chars_inserted", 0)
        output_chars = call.get("output_chars", 0)

        current_request = extract_current_request(user_prompt)

        call_rows.append({
            "timestamp": call.get("timestamp"),
            "run_id": call.get("run_id"),
            "call_type": call.get("call_type"),
            "thought_seconds": call.get("thought_seconds"),
            "thought_weight": call.get("thought_weight"),
            "prompt_chars": prompt_chars,
            "output_chars": output_chars,
            "memory_piece_count_inserted": metadata.get("memory_piece_count_inserted"),
            "memory_chars_inserted": memory_chars,
            "memory_fraction_of_prompt": memory_chars / prompt_chars if prompt_chars else None,
            "output_to_prompt_ratio": output_chars / prompt_chars if prompt_chars else None,
            "memory_truncated_by_char_cap": metadata.get("memory_truncated_by_char_cap"),
            "has_memory_block": has_memory_block(user_prompt),
            "parsed_memory_items": len(memory_items),
            "system_prompt_constant_text": call.get("system_prompt", "").strip(),
            "current_request_chars": len(current_request),
            "output_preview": output[:300].replace("\n", " "),
        })

    source_run_counts = Counter(row.get("source_run_id", "unknown") for row in memory_rows)
    source_type_counts = Counter(row.get("type", "unknown") for row in memory_rows)
    current_call_counts = Counter(row.get("current_call_type", "unknown") for row in memory_rows)

    repeated_memory_text_signals = Counter()

    for call in calls:
        text = call.get("user_prompt", "")
        for phrase in [
            "Alexei Petrovich",
            "sold his books",
            "sold his books, his clothes, his very soul",
            "shadow of the man he once was",
            "conscience, a relentless tormentor",
            "debts gnawed at him",
            "dim light of the alleyway",
        ]:
            if phrase in text:
                repeated_memory_text_signals[phrase] += 1

    system_prompts = [row["system_prompt_constant_text"] for row in call_rows]
    unique_system_prompts = sorted(set(system_prompts))

    stable_fields = {
        "run_id": data.get("run_id"),
        "total_calls_reported": data.get("total_calls"),
        "max_new_tokens_values": sorted(set(
            call.get("metadata", {}).get("max_new_tokens")
            for call in calls
        )),
        "temperature_values": sorted(set(
            str(call.get("metadata", {}).get("temperature"))
            for call in calls
        )),
        "memory_piece_count_values": sorted(set(
            call.get("metadata", {}).get("memory_piece_count_inserted")
            for call in calls
        )),
        "memory_chars_inserted_values": sorted(set(
            call.get("metadata", {}).get("memory_chars_inserted")
            for call in calls
        )),
        "memory_truncation_values": sorted(set(
            str(call.get("metadata", {}).get("memory_truncated_by_char_cap"))
            for call in calls
        )),
        "unique_system_prompt_count": len(unique_system_prompts),
    }

    stats = {
        "thought_seconds": numeric_summary([r["thought_seconds"] for r in call_rows]),
        "thought_weight": numeric_summary([r["thought_weight"] for r in call_rows]),
        "prompt_chars": numeric_summary([r["prompt_chars"] for r in call_rows]),
        "output_chars": numeric_summary([r["output_chars"] for r in call_rows]),
        "memory_chars_inserted": numeric_summary([r["memory_chars_inserted"] for r in call_rows]),
        "memory_fraction_of_prompt": numeric_summary([r["memory_fraction_of_prompt"] for r in call_rows]),
        "output_to_prompt_ratio": numeric_summary([r["output_to_prompt_ratio"] for r in call_rows]),
        "memory_retention_ratio": numeric_summary([r.get("retention_ratio") for r in memory_rows]),
        "memory_thought_seconds_reused": numeric_summary([r.get("thought_seconds") for r in memory_rows]),
        "memory_weight_reused": numeric_summary([r.get("weight") for r in memory_rows]),
        "retained_chars": numeric_summary([r.get("retained_chars") for r in memory_rows]),
        "original_output_chars": numeric_summary([r.get("original_output_chars") for r in memory_rows]),
    }

    return {
        "generated_at": datetime.now().isoformat(),
        "run_id": data.get("run_id"),
        "reported": {
            "total_calls": data.get("total_calls"),
            "total_thought_seconds": data.get("total_thought_seconds"),
            "average_thought_seconds": data.get("average_thought_seconds"),
            "min_thought_seconds": data.get("min_thought_seconds"),
            "max_thought_seconds": data.get("max_thought_seconds"),
            "total_thought_weight": data.get("total_thought_weight"),
            "total_prompt_chars": data.get("total_prompt_chars"),
            "total_output_chars": data.get("total_output_chars"),
            "memory_composition_summary": data.get("memory_composition_summary", {}),
        },
        "computed": {
            "calls_found": len(call_rows),
            "memory_items_found": len(memory_rows),
            "source_run_counts": dict(source_run_counts),
            "source_type_counts": dict(source_type_counts),
            "current_call_memory_counts": dict(current_call_counts),
            "repeated_memory_text_signals": dict(repeated_memory_text_signals),
            "stable_fields": stable_fields,
            "unique_system_prompts": unique_system_prompts,
            "statistics": stats,
        },
        "call_rows": call_rows,
        "memory_rows": memory_rows,
    }


def write_summary_report(analysis: dict, path: str):
    reported = analysis["reported"]
    computed = analysis["computed"]
    stats = computed["statistics"]
    memory_summary = reported.get("memory_composition_summary", {})

    lines = []

    lines.append("# Memory Thought Summary Report\n")
    lines.append(f"Generated: `{analysis['generated_at']}`")
    lines.append(f"Run ID: `{analysis['run_id']}`\n")

    lines.append("## High-Level Run Statistics\n")
    lines.append(f"- Reported total calls: `{reported.get('total_calls')}`")
    lines.append(f"- Calls found in file: `{computed['calls_found']}`")
    lines.append(f"- Reported total thought seconds: `{reported.get('total_thought_seconds')}`")
    lines.append(f"- Reported average thought seconds: `{reported.get('average_thought_seconds')}`")
    lines.append(f"- Reported prompt characters: `{reported.get('total_prompt_chars')}`")
    lines.append(f"- Reported output characters: `{reported.get('total_output_chars')}`")
    lines.append("")

    lines.append("## Memory Composition\n")
    lines.append(f"- Composition records: `{memory_summary.get('composition_records')}`")
    lines.append(f"- Total inserted memory pieces: `{memory_summary.get('total_inserted_piece_count')}`")
    lines.append(f"- Total inserted memory characters: `{memory_summary.get('total_inserted_memory_chars')}`")
    lines.append(f"- Average memory fraction of augmented prompt: `{memory_summary.get('average_memory_fraction_of_augmented_prompt')}`")
    lines.append(f"- Memory items parsed from prompts: `{computed['memory_items_found']}`")
    lines.append("")

    lines.append("## Computed Memory Fraction\n")
    s = stats["memory_fraction_of_prompt"]
    lines.append(f"- Mean memory fraction: `{fmt(s['mean'])}`")
    lines.append(f"- Min memory fraction: `{fmt(s['min'])}`")
    lines.append(f"- Max memory fraction: `{fmt(s['max'])}`")
    lines.append("")

    lines.append("## Thought-Time Statistics\n")
    s = stats["thought_seconds"]
    lines.append(f"- Mean call thought seconds: `{fmt(s['mean'])}`")
    lines.append(f"- Min call thought seconds: `{fmt(s['min'])}`")
    lines.append(f"- Max call thought seconds: `{fmt(s['max'])}`")
    lines.append(f"- Total call thought seconds: `{fmt(s['sum'])}`")
    lines.append("")

    lines.append("## Reused Memory Thought-Time\n")
    s = stats["memory_thought_seconds_reused"]
    lines.append(f"- Reused memory thought seconds total: `{fmt(s['sum'])}`")
    lines.append(f"- Reused memory thought seconds mean per memory item: `{fmt(s['mean'])}`")
    lines.append("")

    lines.append("## Stable Aspects\n")
    stable = computed["stable_fields"]
    for k, v in stable.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## Source Run Counts\n")
    for k, v in computed["source_run_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Repeated Surface Memory Signals\n")
    for k, v in computed["repeated_memory_text_signals"].items():
        lines.append(f"- `{k}` appeared in {v} prompt(s)")
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "This run shows persistent memory being inserted into every LLM call. "
        "The memory fraction indicates how much of the active prompt came from prior-run memory. "
        "Stable values such as fixed memory piece count and fixed inserted memory characters suggest "
        "the memory system is using a consistent injection policy. Repeated surface signals reveal "
        "whether the memory retained literal prose content rather than only abstract style features."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_detailed_report(analysis: dict, path: str):
    lines = []

    lines.append("# Detailed Memory Thought Report\n")
    lines.append(f"Generated: `{analysis['generated_at']}`")
    lines.append(f"Run ID: `{analysis['run_id']}`\n")

    lines.append("## Per-Call Report\n")
    lines.append(
        "| Call | Thought sec | Weight | Prompt chars | Output chars | Memory chars | Memory fraction | Pieces | Truncated |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in analysis["call_rows"]:
        lines.append(
            f"| {row['call_type']} "
            f"| {fmt(row['thought_seconds'])} "
            f"| {fmt(row['thought_weight'])} "
            f"| {fmt(row['prompt_chars'])} "
            f"| {fmt(row['output_chars'])} "
            f"| {fmt(row['memory_chars_inserted'])} "
            f"| {fmt(row['memory_fraction_of_prompt'])} "
            f"| {fmt(row['memory_piece_count_inserted'])} "
            f"| {row['memory_truncated_by_char_cap']} |"
        )

    lines.append("\n## Parsed Memory Items\n")
    lines.append(
        "| Current call | Memory item | Source run | Source type | Thought sec | Weight | Retained chars | Original chars | Retention ratio |"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|")

    for row in analysis["memory_rows"]:
        lines.append(
            f"| {row.get('current_call_type')} "
            f"| {row.get('memory_item_number')} "
            f"| {row.get('source_run_id')} "
            f"| {row.get('type')} "
            f"| {fmt(row.get('thought_seconds'))} "
            f"| {fmt(row.get('weight'))} "
            f"| {fmt(row.get('retained_chars'))} "
            f"| {fmt(row.get('original_output_chars'))} "
            f"| {fmt(row.get('retention_ratio'))} |"
        )

    lines.append("\n## System Prompt Stability\n")
    lines.append(f"Unique system prompts: `{len(analysis['computed']['unique_system_prompts'])}`\n")

    for i, prompt in enumerate(analysis["computed"]["unique_system_prompts"], start=1):
        preview = prompt[:800].replace("\n", " ")
        lines.append(f"### System prompt {i}\n")
        lines.append(f"`{preview}`\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return

    keys = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory_thought_report.json and produce summary/detailed reports."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to memory_thought_report.json"
    )

    parser.add_argument(
        "--output_dir",
        default="memory_thought_report_analysis",
        help="Directory for output reports"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_json(args.input)
    analysis = analyze(data)

    summary_md = os.path.join(args.output_dir, "memory_summary_report.md")
    detailed_md = os.path.join(args.output_dir, "memory_detailed_report.md")
    stats_json = os.path.join(args.output_dir, "memory_statistics.json")
    calls_csv = os.path.join(args.output_dir, "call_statistics.csv")
    memory_csv = os.path.join(args.output_dir, "memory_items.csv")

    write_summary_report(analysis, summary_md)
    write_detailed_report(analysis, detailed_md)
    write_csv(calls_csv, analysis["call_rows"])
    write_csv(memory_csv, analysis["memory_rows"])

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {summary_md}")
    print(f"Wrote: {detailed_md}")
    print(f"Wrote: {stats_json}")
    print(f"Wrote: {calls_csv}")
    print(f"Wrote: {memory_csv}")


if __name__ == "__main__":
    main()
