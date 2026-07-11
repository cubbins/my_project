#!/usr/bin/env bash
set -euo pipefail

# run_nvdisasm_reports.sh
#
# Generate a "maximal reporting" set of nvdisasm outputs for a cubin.
#
# Usage:
#   chmod +x run_nvdisasm_reports.sh
#   ./run_nvdisasm_reports.sh fastWalshTransform_debug.cubin
#
# Optional:
#   ./run_nvdisasm_reports.sh fastWalshTransform_debug.cubin output_dir
#
# Outputs:
#   nvdisasm_result.json
#   nvdisasm_result_src.txt
#   nvdisasm_result_liverange.txt
#   nvdisasm_cfg_hyperblock.dot
#   nvdisasm_cfg_basicblock.dot
#
# Notes:
# - JSON output is generated separately from text/CFG modes.
# - Source-line and live-range reporting are split because life-range mode
#   can override/ignore line-info formatting behavior.
# - CFG outputs are Graphviz .dot files.

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <cubin_file> [output_dir]" >&2
    exit 1
fi

CUBIN_FILE="$1"
OUTDIR="${2:-nvdisasm_reports}"

if [[ ! -f "$CUBIN_FILE" ]]; then
    echo "Error: cubin file not found: $CUBIN_FILE" >&2
    exit 1
fi

if ! command -v nvdisasm >/dev/null 2>&1; then
    echo "Error: nvdisasm not found in PATH" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

BASENAME="$(basename "$CUBIN_FILE")"
STEM="${BASENAME%.*}"

JSON_OUT="$OUTDIR/${STEM}_nvdisasm_result.json"
SRC_OUT="$OUTDIR/${STEM}_nvdisasm_result_src.txt"
LIVERANGE_OUT="$OUTDIR/${STEM}_nvdisasm_result_liverange.txt"
CFG_HB_OUT="$OUTDIR/${STEM}_nvdisasm_cfg_hyperblock.dot"
CFG_BB_OUT="$OUTDIR/${STEM}_nvdisasm_cfg_basicblock.dot"
VERSION_OUT="$OUTDIR/${STEM}_nvdisasm_version.txt"

echo "==> Writing version information"
nvdisasm --version > "$VERSION_OUT"

echo "==> Writing JSON disassembly"
nvdisasm \
  --emit-json \
  --print-code \
  --sort-sections \
  "$CUBIN_FILE" \
  > "$JSON_OUT"

echo "==> Writing source-oriented text disassembly"
nvdisasm \
  --print-code \
  --separate-functions \
  --sort-sections \
  --print-instruction-encoding \
  --print-line-info-inline \
  "$CUBIN_FILE" \
  > "$SRC_OUT"

echo "==> Writing live-range-oriented text disassembly"
nvdisasm \
  --print-code \
  --separate-functions \
  --sort-sections \
  --print-instruction-encoding \
  --print-life-ranges \
  --life-range-mode wide \
  "$CUBIN_FILE" \
  > "$LIVERANGE_OUT"

echo "==> Writing hyperblock CFG"
nvdisasm \
  --output-control-flow-graph \
  --print-instr-offsets-cfg \
  "$CUBIN_FILE" \
  > "$CFG_HB_OUT"

echo "==> Writing basic-block CFG"
nvdisasm \
  --output-control-flow-graph-with-basic-blocks \
  --print-instr-offsets-cfg \
  "$CUBIN_FILE" \
  > "$CFG_BB_OUT"

echo
echo "Done."
echo "Generated files:"
echo "  $VERSION_OUT"
echo "  $JSON_OUT"
echo "  $SRC_OUT"
echo "  $LIVERANGE_OUT"
echo "  $CFG_HB_OUT"
echo "  $CFG_BB_OUT"
