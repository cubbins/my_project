#!/usr/bin/env python3
"""
Analyze memory_composition_report.json and write detailed statistical reports.

Example:
python analyze_memory_composition_report.py \
  --input ./dostoevsky_influenced_run/2026-07-07_12-04-42/memory_composition_report.json \
  --output_dir memory_report_analysis

python analyze_memory_composition_report.py \
  --input ./dostoevsky_influenced_run/2026-07-07_12-04-42/memory_composition_report.json \
  --output_dir memory_composition_analysis


"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def numeric_summary(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "sum": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sum": sum(values),
    }


def fmt_number(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def extract_calls(data: Any) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("calls"), list):
        return data["calls"]

    if isinstance(data, list):
        return data

    return []


def extract_memory_items_from_prompt(prompt: str) -> list[dict]:
    """
    Extract memory item header lines from the stored prompt text.

    Example header:
    [Memory item 3; type=llm_call_4; time=...; source_run_id=...;
     thought_seconds=21.711; weight=66.024; retained_chars=900;
     original_output_chars=3037]
    """
    import re

    pattern = re.compile(r"\[Memory item (?P<item>\d+); (?P<body>[^\]]+)\]")
    items = []

    for match in pattern.finditer(prompt or ""):
        body = match.group("body")
        fields = {"memory_item": int(match.group("item"))}

        for part in body.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key in {"thought_seconds", "weight"}:
                try:
                    fields[key] = float(value)
                except ValueError:
                    fields[key] = value
            elif key in {"retained_chars", "original_output_chars"}:
                try:
                    fields[key] = int(value)
                except ValueError:
                    fields[key] = value
            else:
                fields[key] = value

        items.append(fields)

    return items


def analyze_report(data: Any) -> dict:
    calls = extract_calls(data)

    flattened_top = flatten_dict(data) if isinstance(data, dict) else {}

    call_type_counter = Counter()
    system_prompt_counter = Counter()
    source_run_counter = Counter()
    memory_type_counter = Counter()

    thought_seconds = []
    thought_weights = []
    prompt_chars = []
    output_chars = []
    memory_chars_inserted = []
    memory_piece_counts = []

    all_memory_items = []

    for call in calls:
        call_type = call.get("call_type", "unknown")
        call_type_counter[call_type] += 1

        system_prompt = call.get("system_prompt", "")
        system_prompt_counter[system_prompt] += 1

        if isinstance(call.get("thought_seconds"), (int, float)):
            thought_seconds.append(float(call["thought_seconds"]))

        if isinstance(call.get("thought_weight"), (int, float)):
            thought_weights.append(float(call["thought_weight"]))

        if isinstance(call.get("prompt_chars"), (int, float)):
            prompt_chars.append(float(call["prompt_chars"]))

        if isinstance(call.get("output_chars"), (int, float)):
            output_chars.append(float(call["output_chars"]))

        metadata = call.get("metadata", {})
        if isinstance(metadata, dict):
            if isinstance(metadata.get("memory_chars_inserted"), (int, float)):
                memory_chars_inserted.append(float(metadata["memory_chars_inserted"]))

            if isinstance(metadata.get("memory_piece_count_inserted"), (int, float)):
                memory_piece_counts.append(float(metadata["memory_piece_count_inserted"]))

        items = extract_memory_items_from_prompt(call.get("user_prompt", ""))
        all_memory_items.extend(items)

        for item in items:
            source_run_counter[item.get("source_run_id", "unknown")] += 1
            memory_type_counter[item.get("type", "unknown")] += 1

    constant_fields = {}
    variable_fields = defaultdict(set)

    for call in calls:
        flat = flatten_dict(call)
        for k, v in flat.items():
            try:
                hash(v)
                variable_fields[k].add(v)
            except TypeError:
                variable_fields[k].add(json.dumps(v, sort_keys=True))

    for k, values in variable_fields.items():
        if len(values) == 1:
            constant_fields[k] = next(iter(values))

    variable_field_summary = {
        k: len(v)
        for k, v in sorted(variable_fields.items())
        if len(v) > 1
    }

    memory_retention_ratios = []
    for item in all_memory_items:
        retained = item.get("retained_chars")
        original = item.get("original_output_chars")
        if isinstance(retained, int) and isinstance(original, int) and original > 0:
            memory_retention_ratios.append(retained / original)

    return {
        "generated_at": datetime.now().isoformat(),
        "top_level_fields": sorted(flattened_top.keys()),
        "top_level_summary": {
            "run_id": data.get("run_id") if isinstance(data, dict) else None,
            "total_calls": len(calls),
            "reported_total_calls": data.get("total_calls") if isinstance(data, dict) else None,
            "reported_total_thought_seconds": data.get("total_thought_seconds") if isinstance(data, dict) else None,
            "reported_average_thought_seconds": data.get("average_thought_seconds") if isinstance(data, dict) else None,
        },
        "call_statistics": {
            "thought_seconds": numeric_summary(thought_seconds),
            "thought_weight": numeric_summary(thought_weights),
            "prompt_chars": numeric_summary(prompt_chars),
            "output_chars": numeric_summary(output_chars),
            "memory_chars_inserted": numeric_summary(memory_chars_inserted),
            "memory_piece_count_inserted": numeric_summary(memory_piece_counts),
        },
        "memory_item_statistics": {
            "total_memory_items_extracted": len(all_memory_items),
            "source_run_counts": dict(source_run_counter),
            "memory_type_counts": dict(memory_type_counter),
            "retention_ratio": numeric_summary(memory_retention_ratios),
        },
        "constant_call_fields": constant_fields,
        "variable_call_field_unique_counts": variable_field_summary,
        "call_type_counts": dict(call_type_counter),
        "system_prompt_counts": dict(system_prompt_counter),
        "memory_items": all_memory_items,
    }


def write_markdown_report(analysis: dict, output_path: str):
    top = analysis["top_level_summary"]
    stats = analysis["call_statistics"]
    mem = analysis["memory_item_statistics"]

    lines = []

    lines.append("# Memory Composition Analysis Report\n")
    lines.append(f"Generated at: `{analysis['generated_at']}`\n")
    lines.append("## Run Summary\n")
    lines.append(f"- Run ID: `{top.get('run_id')}`")
    lines.append(f"- Calls found: `{top.get('total_calls')}`")
    lines.append(f"- Reported calls: `{top.get('reported_total_calls')}`")
    lines.append(f"- Reported total thought seconds: `{top.get('reported_total_thought_seconds')}`")
    lines.append(f"- Reported average thought seconds: `{top.get('reported_average_thought_seconds')}`")
    lines.append("")

    lines.append("## Call Statistics\n")
    for name, summary in stats.items():
        lines.append(f"### {name}")
        lines.append(f"- Count: {summary['count']}")
        lines.append(f"- Min: {fmt_number(summary['min'])}")
        lines.append(f"- Max: {fmt_number(summary['max'])}")
        lines.append(f"- Mean: {fmt_number(summary['mean'])}")
        lines.append(f"- Median: {fmt_number(summary['median'])}")
        lines.append(f"- Sum: {fmt_number(summary['sum'])}")
        lines.append("")

    lines.append("## Memory Item Statistics\n")
    lines.append(f"- Total memory items extracted from prompts: `{mem['total_memory_items_extracted']}`")
    lines.append("")
    lines.append("### Source run counts")
    for k, v in mem["source_run_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Memory type counts")
    for k, v in mem["memory_type_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    rr = mem["retention_ratio"]
    lines.append("### Retention ratio")
    lines.append("- Retention ratio = retained_chars / original_output_chars")
    lines.append(f"- Count: {rr['count']}")
    lines.append(f"- Min: {fmt_number(rr['min'])}")
    lines.append(f"- Max: {fmt_number(rr['max'])}")
    lines.append(f"- Mean: {fmt_number(rr['mean'])}")
    lines.append(f"- Median: {fmt_number(rr['median'])}")
    lines.append("")

    lines.append("## Call Type Counts\n")
    for k, v in analysis["call_type_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Constant Call Fields\n")
    lines.append("These fields had exactly one value across all calls.\n")
    for k, v in analysis["constant_call_fields"].items():
        text = str(v)
        if len(text) > 300:
            text = text[:300] + "..."
        lines.append(f"- `{k}`: `{text}`")
    lines.append("")

    lines.append("## Variable Call Field Unique Counts\n")
    lines.append("These fields changed across calls.\n")
    for k, v in analysis["variable_call_field_unique_counts"].items():
        lines.append(f"- `{k}`: {v} unique values")
    lines.append("")

    lines.append("## Extracted Memory Items\n")
    for item in analysis["memory_items"]:
        lines.append(
            f"- Memory item {item.get('memory_item')}: "
            f"type=`{item.get('type')}`, "
            f"source_run_id=`{item.get('source_run_id')}`, "
            f"thought_seconds=`{item.get('thought_seconds')}`, "
            f"weight=`{item.get('weight')}`, "
            f"retained_chars=`{item.get('retained_chars')}`, "
            f"original_output_chars=`{item.get('original_output_chars')}`"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_csv_memory_items(memory_items: list[dict], output_path: str):
    import csv

    fields = [
        "memory_item",
        "type",
        "time",
        "source_run_id",
        "thought_seconds",
        "weight",
        "retained_chars",
        "original_output_chars",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in memory_items:
            writer.writerow({k: item.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory_composition_report.json and produce detailed reports."
    )

    parser.add_argument(
        "--input",
        default="./dostoevsky_influenced_run/2026-07-07_12-04-42/memory_composition_report.json",
        help="Path to memory_composition_report.json"
    )

    parser.add_argument(
        "--output_dir",
        default="memory_composition_analysis",
        help="Directory to write analysis outputs"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_json(args.input)
    analysis = analyze_report(data)

    json_out = os.path.join(args.output_dir, "memory_composition_analysis.json")
    md_out = os.path.join(args.output_dir, "memory_composition_analysis.md")
    csv_out = os.path.join(args.output_dir, "memory_items.csv")

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    write_markdown_report(analysis, md_out)
    write_csv_memory_items(analysis["memory_items"], csv_out)

    print(f"Wrote: {json_out}")
    print(f"Wrote: {md_out}")
    print(f"Wrote: {csv_out}")


if __name__ == "__main__":
    main()
