#!/usr/bin/env python3
import json
import re
import sys
from collections import defaultdict

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

STEP_CMD_RE = re.compile(r"^\(cuda-gdb\)\s+stepi\s*$")
INFO_REGS_RE = re.compile(r"^\(cuda-gdb\)\s+info registers\s*$")
INFO_REGS_PC_RE = re.compile(r"^\(cuda-gdb\)\s+info registers pc\s*$")
X5I_RE = re.compile(r"^\(cuda-gdb\)\s+x/5i \$pc\s*$")

STEPI_RESULT_RE = re.compile(r"^(0x[0-9a-fA-F]+)\s+(\d+)\s+(.*)$")
PC_LINE_RE = re.compile(r"^pc\s+(0x[0-9a-fA-F]+)")
REG_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s+(\S+)\s+(.*)$")
INSTR_LINE_RE = re.compile(
    r"^\s*(=>\s*)?(0x[0-9a-fA-F]+)\s+<[^>]+>:\s+([A-Z0-9_.]+)\s*(.*)$"
)

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)

def parse_log(text: str):
    text = strip_ansi(text)
    lines = [line.rstrip("\n") for line in text.splitlines()]

    report = {
        "summary": {
            "kernel": None,
            "source_file": None,
            "entry_source_line": None,
            "step_count": 0,
            "changed_registers_ranked": [],
            "mostly_static_registers_ranked": []
        },
        "steps": []
    }

    for line in lines:
        if "CUDA thread hit application kernel entry function breakpoint" in line:
            m = re.search(r"breakpoint,\s*([A-Za-z0-9_]+)", line)
            if m:
                report["summary"]["kernel"] = m.group(1)

        if line.startswith("Current source file is "):
            report["summary"]["source_file"] = line[len("Current source file is "):]

        m = re.search(r"at\s+(.+):(\d+)$", line)
        if m and "fastWalshTransform_kernel.cuh" in line:
            report["summary"]["entry_source_line"] = int(m.group(2))

    i = 0
    while i < len(lines):
        line = lines[i]

        if STEP_CMD_RE.match(line):
            step = {
                "step": len(report["steps"]) + 1,
                "pc": None,
                "source_line": None,
                "source_text": None,
                "registers": {},
                "instruction_window": [],
                "changed": {}
            }

            if i + 1 < len(lines):
                m = STEPI_RESULT_RE.match(lines[i + 1].strip())
                if m:
                    step["pc"] = m.group(1)
                    step["source_line"] = int(m.group(2))
                    step["source_text"] = m.group(3)

            j = i + 1
            while j < len(lines):
                s = lines[j]

                if j > i and STEP_CMD_RE.match(s):
                    break

                if INFO_REGS_RE.match(s):
                    k = j + 1
                    while k < len(lines):
                        t = lines[k].strip()
                        if t.startswith("(cuda-gdb)"):
                            break
                        m = REG_LINE_RE.match(t)
                        if m:
                            step["registers"][m.group(1)] = {
                                "value": m.group(2),
                                "extra": m.group(3).strip()
                            }
                        k += 1
                    j = k
                    continue

                if INFO_REGS_PC_RE.match(s):
                    k = j + 1
                    while k < len(lines):
                        t = lines[k].strip()
                        if t.startswith("(cuda-gdb)"):
                            break
                        m = PC_LINE_RE.match(t)
                        if m:
                            step["pc"] = m.group(1)
                            break
                        k += 1
                    j = k
                    continue

                if X5I_RE.match(s):
                    k = j + 1
                    while k < len(lines):
                        t = lines[k].rstrip()
                        if t.startswith("(cuda-gdb)"):
                            break
                        m = INSTR_LINE_RE.match(t)
                        if m:
                            step["instruction_window"].append({
                                "current": bool(m.group(1)),
                                "addr": m.group(2),
                                "opcode": m.group(3),
                                "operands": m.group(4).strip()
                            })
                        k += 1
                    j = k
                    continue

                j += 1

            report["steps"].append(step)
            i = j
            continue

        i += 1

    prev_regs = None
    changed_counts = defaultdict(int)
    static_counts = defaultdict(int)

    for step in report["steps"]:
        regs = step["registers"]
        if prev_regs is not None:
            all_names = sorted(set(prev_regs) | set(regs))
            for reg in all_names:
                old = prev_regs.get(reg, {}).get("value")
                new = regs.get(reg, {}).get("value")
                if old != new:
                    step["changed"][reg] = {"old": old, "new": new}
                    changed_counts[reg] += 1
                else:
                    static_counts[reg] += 1
        prev_regs = regs

    report["summary"]["step_count"] = len(report["steps"])
    report["summary"]["changed_registers_ranked"] = sorted(
        changed_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )
    report["summary"]["mostly_static_registers_ranked"] = sorted(
        static_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )

    return report

def write_markdown(report, path):
    with open(path, "w", encoding="utf-8") as f:
        s = report["summary"]
        f.write("# CUDA-GDB Step Analysis Report\n\n")
        f.write(f"- Kernel: `{s['kernel']}`\n")
        f.write(f"- Source file: `{s['source_file']}`\n")
        f.write(f"- Entry source line: `{s['entry_source_line']}`\n")
        f.write(f"- Steps parsed: `{s['step_count']}`\n\n")

        f.write("## Registers that changed most often\n\n")
        f.write("| Register | Count |\n|---|---:|\n")
        for reg, cnt in s["changed_registers_ranked"][:20]:
            f.write(f"| {reg} | {cnt} |\n")
        f.write("\n")

        for step in report["steps"]:
            f.write(f"## Step {step['step']}\n\n")
            f.write(f"- PC: `{step['pc']}`\n")
            f.write(f"- Source line: `{step['source_line']}`\n")
            f.write(f"- Source text: `{step['source_text']}`\n")
            if step["instruction_window"]:
                cur = next((x for x in step["instruction_window"] if x["current"]), step["instruction_window"][0])
                f.write(f"- Current instruction: `{cur['opcode']} {cur['operands']}`\n")
            f.write("\n")
            if step["changed"]:
                f.write("| Register | Old | New |\n|---|---|---|\n")
                for reg, delta in sorted(step["changed"].items()):
                    f.write(f"| {reg} | `{delta['old']}` | `{delta['new']}` |\n")
            else:
                f.write("No deltas recorded for this step.\n")
            f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("usage: python3 analyze_cuda_gdb_steps_log.py cuda_gdb_pty_steps10_ops.log")
        sys.exit(1)

    log_path = sys.argv[1]

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    report = parse_log(text)

    json_path = "cuda_gdb_steps_analysis.json"
    md_path = "cuda_gdb_steps_analysis.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    write_markdown(report, md_path)

    print(f"Parsed steps: {report['summary']['step_count']}")
    print(f"Kernel: {report['summary']['kernel']}")
    print(f"Source file: {report['summary']['source_file']}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

if __name__ == "__main__":
    main()