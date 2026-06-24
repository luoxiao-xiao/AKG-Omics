#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

BASE_RUN_TAG="${RUN_TAG:-proteogenomics_rule_agent_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
FINAL_SEEDS="${FINAL_SEEDS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-700}"
COMMON_SAVE_PARENT="${SAVE_PARENT:-${PROJECT_ROOT}/results/akg_omics_proteogenomics}"
mkdir -p "${COMMON_SAVE_PARENT}"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEVICE FINAL_SEEDS TRAIN_EPOCHS

export RUN_PROTEOGENOMICS_TASKS="${RUN_PROTEOGENOMICS_TASKS:-1}"
export RUN_METABOLOMICS_TASKS="${RUN_METABOLOMICS_TASKS:-0}"
export PROTEO_DATA_ROOT="${PROTEO_DATA_ROOT:-${PROJECT_ROOT}/data/spatch}"
export PROTEO_SAMPLE1="${PROTEO_SAMPLE1:-protein}"
export PROTEO_SAMPLE2="${PROTEO_SAMPLE2:-gene}"

export TRAIN_LR="${TRAIN_LR:-1e-3}"
export TRAIN_HIDDEN_DIM="${TRAIN_HIDDEN_DIM:-256}"
export TRAIN_DROPOUT="${TRAIN_DROPOUT:-0.1}"
export PROTEO_GENE_LATENT_DIM="${PROTEO_GENE_LATENT_DIM:-128}"

export LAMBDA_TARGET="${LAMBDA_TARGET:-1.0}"
export LAMBDA_PCC="${LAMBDA_PCC:-0.0}"
export LAMBDA_LATENT="${LAMBDA_LATENT:-0.5}"
export LAMBDA_RECON="${LAMBDA_RECON:-0.2}"
export LAMBDA_KB="${LAMBDA_KB:-0.05}"
export LAMBDA_GRAPH="${LAMBDA_GRAPH:-0.05}"
export LAMBDA_ALIGN="${LAMBDA_ALIGN:-0.02}"
export LAMBDA_FULL="${LAMBDA_FULL:-1.0}"
export LAMBDA_PARTIAL="${LAMBDA_PARTIAL:-0.0}"
export TARGET_SOFT_ALPHA="${TARGET_SOFT_ALPHA:-0.0}"
export KB_CT_PRIOR_MIX="${KB_CT_PRIOR_MIX:-0.7}"
export KB_USAGE_MODE="${KB_USAGE_MODE:-warmup_and_injection}"

export EARLY_STOP="${EARLY_STOP:-1}"
export EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-loss_total}"
export EARLY_STOP_MIN_EPOCHS="${EARLY_STOP_MIN_EPOCHS:-0}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-30}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
export EARLY_STOP_RESTORE_BEST="${EARLY_STOP_RESTORE_BEST:-0}"

export FORCE_TASK_RETRAIN="${FORCE_TASK_RETRAIN:-1}"
export USE_KB_QUALITY_SCALING="${USE_KB_QUALITY_SCALING:-0}"
export DISABLE_DYNAMIC_RETRIEVER="${DISABLE_DYNAMIC_RETRIEVER:-0}"
export DISABLE_KB_FUSION="${DISABLE_KB_FUSION:-0}"
export DISABLE_SOFT_GATE="${DISABLE_SOFT_GATE:-1}"
export DISABLE_RESIDUAL_REFINE="${DISABLE_RESIDUAL_REFINE:-0}"

run_one_mode() {
  local mode_name="$1"
  local run_name="$2"
  local kb_agent="$3"
  local requested_mode="$4"
  local data_profile="$5"
  local require_agent="$6"
  local agent_enabled="$7"
  local save_root="${COMMON_SAVE_PARENT}/${BASE_RUN_TAG}/${mode_name}"
  mkdir -p "${save_root}"

  export RUN_NAME="${run_name}"
  export SAVE_ROOT="${save_root}"
  export RUN_TAG="${BASE_RUN_TAG}_${mode_name}"
  export PROTEO_TASK_CACHE_VERSION="${PROTEO_TASK_CACHE_VERSION_PREFIX:-akg_omics}_${BASE_RUN_TAG}_${mode_name}"
  export USE_PROTEO_KB_AGENT="${kb_agent}"
  export REQUIRE_PROTEO_KB_LLM_AGENT="${require_agent}"
  export KNOWLEDGE_AGENT_ENABLE_AGENT="${agent_enabled}"
  export PROTEO_KB_REQUESTED_MODE="${requested_mode}"
  export PROTEO_KB_SELECTION_MODE="${requested_mode}"
  export PROTEO_KB_USE_DATA_PROFILE="${data_profile}"

  local log_path="${save_root}/akg_omics_${mode_name}.log"
  echo ">>> Running ${mode_name}"
  echo ">>> Save root: ${save_root}"
  echo ">>> Device: ${DEVICE}; seeds=${FINAL_SEEDS}; max_epochs=${TRAIN_EPOCHS}; patience=${EARLY_STOP_PATIENCE}"
  echo ">>> KB selection: ${PROTEO_KB_REQUESTED_MODE}; data_profile=${PROTEO_KB_USE_DATA_PROFILE}; agent=${KNOWLEDGE_AGENT_ENABLE_AGENT}"
  "${PYTHON_BIN:-python}" -u -m akg_omics.run 2>&1 | tee "${log_path}"
}

run_one_mode "rule" "akg_omics_rule" "0" "rule-only" "0" "0" "0"
run_one_mode "agent" "akg_omics_agent" "1" "agent-rule-data" "1" "0" "1"

echo ">>> Finished AKG-Omics proteogenomics rule and agent runs."
echo ">>> Results root: ${COMMON_SAVE_PARENT}/${BASE_RUN_TAG}"