#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export FINAL_SEEDS="${FINAL_SEEDS:-1}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-700}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEVICE="${DEVICE:-cuda:0}"

export EARLY_STOP=1
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
export EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-loss_total}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
export RUN_PROTEOGENOMICS_TASKS=0
export RUN_METABOLOMICS_TASKS=1
export FORCE_TASK_RETRAIN="${FORCE_TASK_RETRAIN:-1}"

export MET_CLEAN_LOSS=1
export LAMBDA_TARGET=1.0
export LAMBDA_PCC=0
export LAMBDA_LATENT=0
export LAMBDA_RECON=0
export LAMBDA_GRAPH=0
export LAMBDA_ORTHO=0
export LAMBDA_DGI=0
export LAMBDA_BRIDGE_ALIGN=0
export TARGET_SOFT_ALPHA=0
export DISABLE_SOFT_GATE=1

export TRAIN_LR="${TRAIN_LR:-1e-3}"
export TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-256}"
export TRAIN_DROPOUT="${TRAIN_DROPOUT:-0.1}"
export LAMBDA_FULL="${LAMBDA_FULL:-1.0}"
export LAMBDA_PARTIAL="${LAMBDA_PARTIAL:-0.0}"
export KB_USAGE_MODE="${KB_USAGE_MODE:-warmup_and_injection}"
export USE_KB_QUALITY_SCALING=0
export DISABLE_DYNAMIC_RETRIEVER=0
export DISABLE_KB_FUSION=0
export DISABLE_RESIDUAL_REFINE=0

export USE_MET_KB_AGENT=0
export KNOWLEDGE_AGENT_ENABLE_AGENT=0
export REQUIRE_MET_KB_LLM_AGENT=0
export MET_KB_REQUESTED_MODE=rule-only
export MET_KB_SELECTION_MODE=rule-only
export MET_KB_USE_DATA_PROFILE=0
export MET_KB_ALLOWED_SOURCES="hmdb,reactome,chebi"
export MET_KB_SELECTION_POLICY="${MET_KB_SELECTION_POLICY:-akg_omics_metabolomics_final}"
export MET_KB_AGENT_MIN_SOURCES=1
export MET_KB_AGENT_MAX_SOURCES=3
export MET_KB_AGENT_REUSE_SUCCESSFUL_SELECTION=1
export MET_KB_AGENT_STRICT_FAILURE=0
export KNOWLEDGE_AGENT_RAISE_LLM_ERRORS=0
export KNOWLEDGE_AGENT_SAVE_LLM_TRACE=0

configure_task() {
  local task_id="$1"
  local sources="$2"
  local cycle="$3"
  local kb="$4"
  local align="$5"
  local number
  case "${task_id}" in
    task5_*) number=5 ;;
    task6_*) number=6 ;;
    task7_*) number=7 ;;
    task8_*) number=8 ;;
    *) echo "Unknown task: ${task_id}" >&2; return 2 ;;
  esac
  export MET_ONLY_TASKS="${task_id}"
  export MET_KB_ALLOWED_SOURCES="${sources}"
  export "MET_TASK${number}_SOURCES=${sources}"
  export "MET_TASK${number}_LAMBDA_CYCLE=${cycle}"
  export "MET_TASK${number}_LAMBDA_PARTIAL_CYCLE=${cycle}"
  export "MET_TASK${number}_LAMBDA_KB=${kb}"
  export "MET_TASK${number}_LAMBDA_ALIGN=${align}"
  export "MET_TASK${number}_LAMBDA_GRAPH=0"
  export "MET_TASK${number}_LAMBDA_PCC=0"
  export "MET_TASK${number}_DISABLE_SOFT_GATE=1"
  if [[ "${number}" == "7" ]]; then
    export MET_TASK7_HMDB_PPM_TOL=5
    export MET_TASK7_HMDB_TOPK_CANDIDATES=2
    export MET_TASK7_HMDB_MIN_CANDIDATE_WEIGHT=0.02
  fi
}

export ENABLE_DETAILED_ANALYSIS=1
export TOP_FEATURE_N="${TOP_FEATURE_N:-20}"
export PLOT_TOP_FEATURE_N="${PLOT_TOP_FEATURE_N:-5}"
export SAVE_PARENT="${SAVE_PARENT:-${PROJECT_ROOT}/results/akg_omics_metabolomics}"
export BASE_RUN_TAG="${BASE_RUN_TAG:-metabolomics_task5_8_rule_seed${FINAL_SEEDS}_cuda${CUDA_VISIBLE_DEVICES}}"
export SAVE_ROOT="${SAVE_PARENT}/${BASE_RUN_TAG}"
mkdir -p "${SAVE_ROOT}"

TASKS=(
  "task5_he_gene_to_metabolism:10:0:reactome,chebi:0.30:0.05:0.03:task5_reactome_chebi"
  "task6_he_metabolism_to_gene:30:0:reactome,chebi:0.10:0.03:0.02:task6_reactome_chebi"
  "task7_he_to_metabolism:20:1:reactome,hmdb:0.20:0.04:0.03:task7_reactome_hmdb"
  "task8_he_to_gene_in_metabolomics:30:0:reactome:0.30:0.04:0.02:task8_reactome"
)

{
  echo ">>> Running AKG-Omics metabolomics tasks seed=${FINAL_SEEDS}"
  echo ">>> Save root: ${SAVE_ROOT}"
  for spec in "${TASKS[@]}"; do
    IFS=":" read -r task_id patience restore_best sources cycle kb align config_tag <<< "${spec}"
    configure_task "${task_id}" "${sources}" "${cycle}" "${kb}" "${align}"
    export EARLY_STOP_PATIENCE="${patience}"
    export EARLY_STOP_RESTORE_BEST="${restore_best}"
    export RUN_NAME="akg_omics_metabolomics_${task_id}"
    export RUN_TAG="akg_omics_metabolomics_${config_tag}"
    export MET_TASK_CACHE_VERSION="akg_omics_metabolomics_${config_tag}"
    echo ">>> [TASK] ${task_id}: sources=${sources}, cycle=${cycle}, kb=${kb}, align=${align}, patience=${patience}"
    "${PYTHON_BIN}" -u -m akg_omics.run_metabolomics_final 2>&1 | tee "${SAVE_ROOT}/run_${task_id}.log"
  done

  if [[ -f visualization_scripts/make_met_paper_visualizations_v4.py ]]; then
    SAVE_ROOT="${SAVE_ROOT}" PAPER_VIS_TOP_N=5 "${PYTHON_BIN}" -u visualization_scripts/make_met_paper_visualizations_v4.py
  fi

  if [[ "${RUN_BASELINE_VISUALIZATION:-0}" == "1" ]]; then
    if [[ -z "${BASELINE_ROOT:-}" ]]; then
      echo "RUN_BASELINE_VISUALIZATION=1 requires BASELINE_ROOT to point to the baseline repository." >&2
      exit 2
    fi
    BASELINE_CODE_DIR="${BASELINE_CODE_DIR:-${BASELINE_ROOT}}"
    OURS_MANIFEST="${SAVE_ROOT}/paper_visualizations_v4/visualization_manifest.csv"
    SELECTED_JSON="${BASELINE_CODE_DIR}/met_top5_features_for_baselines.json"
    BASELINE_RUN_DIR="${BASELINE_ROOT}/results/baseline_metabolomics_matched_visual_seed1_cuda${CUDA_VISIBLE_DEVICES}"
    BASELINE_VIS_DIR="${BASELINE_RUN_DIR}/paper_visualizations_met_matched_final"
    BIGFIG_DIR="${BASELINE_RUN_DIR}/task_best_prediction_comparison_bigfigs_final"
    "${PYTHON_BIN}" -u "${BASELINE_CODE_DIR}/extract_met_top5_features_for_baselines.py" --manifest "${OURS_MANIFEST}" --out-json "${SELECTED_JSON}" --top-n 5
    "${PYTHON_BIN}" -u "${BASELINE_CODE_DIR}/visualize_baseline_met_matched.py" --run-dir "${BASELINE_RUN_DIR}" --selected-json "${SELECTED_JSON}" --out-dir "${BASELINE_VIS_DIR}" --seed 1
    "${PYTHON_BIN}" -u "${BASELINE_CODE_DIR}/assemble_met_task_best_prediction_comparisons.py" --ours-manifest "${OURS_MANIFEST}" --baseline-manifest "${BASELINE_VIS_DIR}/visualization_manifest.csv" --out-dir "${BIGFIG_DIR}"
  fi
  echo ">>> AKG-Omics metabolomics results: ${SAVE_ROOT}"
} 2>&1 | tee "${SAVE_ROOT}/run.log"