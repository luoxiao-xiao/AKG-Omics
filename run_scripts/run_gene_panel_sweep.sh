#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GENE_DIMS="${GENE_DIMS:-4806 2000 1000 500 300 100 50}"
FINAL_SEEDS="${FINAL_SEEDS:-1,11,111}"
BASE_RUN_TAG="${BASE_RUN_TAG:-proteogenomics_gene_panel_sweep_700_pat30}"
SAVE_PARENT="${SAVE_PARENT:-${PROJECT_ROOT}/results/akg_omics_gene_panel_sweep}"
BENEFICIAL_GENE_LIST="${BENEFICIAL_GENE_LIST:-${PROJECT_ROOT}/gene_selection/beneficial_gene_order_task1_task3_avgpcc_full4806_seed1.csv}"
BENEFICIAL_GENE_TAG="${BENEFICIAL_GENE_TAG:-avgpcc_task1_task3}"
mkdir -p "${SAVE_PARENT}" logs

run_worker() {
  local mode="$1"
  local cuda_id="$2"
  local random_seed="${3:-2026}"

  for gene_dim in ${GENE_DIMS}; do
    local save_root="${SAVE_PARENT}/${BASE_RUN_TAG}/${mode}/gene${gene_dim}"
    mkdir -p "${save_root}"
    (
      export RUN_TAG="${BASE_RUN_TAG}_${mode}_gene${gene_dim}_cuda${cuda_id}"
      export RUN_NAME="akg_omics_gene_panel_${mode}_${gene_dim}_rule"
      export SAVE_ROOT="${save_root}"
      export CUDA_VISIBLE_DEVICES="${cuda_id}"
      export DEVICE=cuda:0
      export RUN_PROTEOGENOMICS_TASKS=1
      export RUN_METABOLOMICS_TASKS=0
      export FINAL_SEEDS="${FINAL_SEEDS}"
      export TRAIN_EPOCHS=700
      export EARLY_STOP=1
      export EARLY_STOP_MIN_EPOCHS=0
      export EARLY_STOP_PATIENCE=30
      export EARLY_STOP_MIN_DELTA=1e-4
      export EARLY_STOP_MONITOR=loss_total
      export EARLY_STOP_RESTORE_BEST=0
      export PROTEO_DATA_ROOT="${PROTEO_DATA_ROOT:-${PROJECT_ROOT}/data/spatch}"
      export PROTEO_SAMPLE1="${PROTEO_SAMPLE1:-protein}"
      export PROTEO_SAMPLE2="${PROTEO_SAMPLE2:-gene}"
      export PROTEO_GENE_TOPK="${gene_dim}"
      export PROTEO_GENE_SELECT_MODE="${mode}"
      export PROTEO_GENE_RANDOM_SEED="${random_seed}"
      export PROTEO_GENE_BENEFICIAL_LIST="${BENEFICIAL_GENE_LIST}"
      export PROTEO_GENE_BENEFICIAL_TAG="${BENEFICIAL_GENE_TAG}"
      export USE_PROTEO_KB_AGENT=0
      export REQUIRE_PROTEO_KB_LLM_AGENT=0
      export KNOWLEDGE_AGENT_ENABLE_AGENT=0
      export PROTEO_KB_REQUESTED_MODE=rule-only
      export PROTEO_KB_SELECTION_MODE=rule-only
      export PROTEO_KB_USE_DATA_PROFILE=0
      export ENABLE_DETAILED_ANALYSIS=0
      echo ">>> [START] mode=${mode} gene_dim=${gene_dim} cuda=${cuda_id} seeds=${FINAL_SEEDS}"
      "${PYTHON_BIN}" -u -m akg_omics.run 2>&1 | tee "${save_root}/akg_omics_${mode}_gene${gene_dim}_rule.log"
    )
  done
}

run_worker beneficial "${BENEFICIAL_CUDA:-0}" 2026 > "logs/${BASE_RUN_TAG}_beneficial.launcher.log" 2>&1 &
pid_beneficial=$!
run_worker random "${RANDOM_CUDA:-1}" 2026 > "logs/${BASE_RUN_TAG}_random.launcher.log" 2>&1 &
pid_random=$!

echo ">>> launched beneficial pid=${pid_beneficial}; random pid=${pid_random}"
echo ">>> result root: ${SAVE_PARENT}/${BASE_RUN_TAG}"
wait "${pid_beneficial}" "${pid_random}"
echo ">>> all gene-panel sweep runs finished."