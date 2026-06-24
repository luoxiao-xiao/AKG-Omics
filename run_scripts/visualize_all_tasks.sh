#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/results/akg_omics_visualization/direct_task12_reexport}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/results/akg_omics_visualization/paper_visualizations}"
MET_ANALYSIS_ROOT="${MET_ANALYSIS_ROOT:-${PROJECT_ROOT}/results/akg_omics_metabolomics/metabolomics_task5_8_rule_seed1_cuda0/detailed_analysis}"
mkdir -p "${EXPORT_ROOT}" "${OUT_ROOT}" logs

export SAVE_ROOT="${EXPORT_ROOT}"
export RUN_NAME="akg_omics_direct_task12_reexport"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda:0}"
export FINAL_SEEDS="${FINAL_SEEDS:-1}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-700}"
export EARLY_STOP=1
export EARLY_STOP_MIN_EPOCHS=0
export EARLY_STOP_PATIENCE=30
export EARLY_STOP_MIN_DELTA=1e-4
export EARLY_STOP_MONITOR=loss_total
export EARLY_STOP_RESTORE_BEST=0
export RUN_PROTEOGENOMICS_TASKS=1
export RUN_METABOLOMICS_TASKS=0
export PROTEO_TASK_FILTER="${PROTEO_TASK_FILTER:-1,2}"
export PROTEO_DATA_ROOT="${PROTEO_DATA_ROOT:-${PROJECT_ROOT}/data/spatch}"
export PROTEO_SAMPLE1="${PROTEO_SAMPLE1:-protein}"
export PROTEO_SAMPLE2="${PROTEO_SAMPLE2:-gene}"
export USE_PROTEO_KB_AGENT=0
export REQUIRE_PROTEO_KB_LLM_AGENT=0
export KNOWLEDGE_AGENT_ENABLE_AGENT=0
export PROTEO_KB_REQUESTED_MODE=rule-only
export PROTEO_KB_SELECTION_MODE=rule-only
export PROTEO_KB_USE_DATA_PROFILE=0
export ENABLE_DETAILED_ANALYSIS=1
export TOP_FEATURE_N="${TOP_FEATURE_N:-4806}"
export PLOT_TOP_FEATURE_N=0

"${PYTHON_BIN}" -u -m akg_omics.run
"${PYTHON_BIN}" -u visualization_scripts/generate_direct_task_visualizations.py \
  --proteo-run-root "${EXPORT_ROOT}" \
  --met-analysis-root "${MET_ANALYSIS_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --threshold "${VIS_THRESHOLD:-0.5}" \
  --top-n-spatial "${VIS_TOP_N_SPATIAL:-5}" \
  --top-n-heatmap "${VIS_TOP_N_HEATMAP:-12}"

echo ">>> Detailed arrays: ${EXPORT_ROOT}/detailed_analysis"
echo ">>> Visualizations: ${OUT_ROOT}"