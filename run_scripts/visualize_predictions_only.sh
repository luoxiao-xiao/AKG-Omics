#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PROTEO_RUN_ROOT="${PROTEO_RUN_ROOT:-${PROJECT_ROOT}/results/akg_omics_proteogenomics/proteogenomics_rule_agent_700_pat30/rule}"
MET_ANALYSIS_ROOT="${MET_ANALYSIS_ROOT:-${PROJECT_ROOT}/results/akg_omics_metabolomics/metabolomics_task5_8_rule_seed1_cuda0/detailed_analysis}"
OUT_ROOT="${OUT_ROOT:-${PROTEO_RUN_ROOT}/paper_visualizations}"

"${PYTHON_BIN}" -u visualization_scripts/generate_direct_task_visualizations.py \
  --proteo-run-root "${PROTEO_RUN_ROOT}" \
  --met-analysis-root "${MET_ANALYSIS_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --threshold "${VIS_THRESHOLD:-0.5}" \
  --top-n-spatial "${VIS_TOP_N_SPATIAL:-5}" \
  --top-n-heatmap "${VIS_TOP_N_HEATMAP:-12}"

echo ">>> Done: ${OUT_ROOT}"