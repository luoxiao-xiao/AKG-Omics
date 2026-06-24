# AKG-Omics

AKG-Omics is a one-for-all knowledge-guided framework for histology-conditioned spatial omics generation. The code supports deterministic rule-based knowledge selection and an optional LLM Agent that selects and adapts external biological knowledge sources before model training.

## Repository layout

```text
akg_omics/                    Core model, training protocols, data preparation, and KB agent
akg_omics/knowledge_agent/    Rule/Agent source selection and LLM client code
run_scripts/                  Reproducible training and visualization entrypoints
analysis_scripts/             External-knowledge ablation and case-study analysis scripts
visualization_scripts/        Figure-oriented spatial prediction visualization scripts
requirements.txt              Python dependency list used in our experiments
pyproject.toml                Editable-install package metadata
```

## 1. Create the environment

We used Python 3.10/3.11 with CUDA-enabled PyTorch. A typical setup is:

```bash
conda create -n akg-omics python=3.10 -y
conda activate akg-omics
pip install -r requirements.txt
pip install -e .
```

If your CUDA/PyTorch versions differ, install the PyTorch build that matches your GPU first, then install the remaining packages.

## 2. Prepare data and knowledge bases

The run scripts expect these paths by default:

```bash
export PROTEO_DATA_ROOT=data/spatch
export PROTEO_KB_DIR=data/KB
export KB_ROOT=data/KB/metabolite
```

For another machine, point these variables to your local data and KB directories before running. The proteogenomics data should contain the paired H&E, protein, and gene inputs used by `akg_omics.run`. The metabolomics KB root should contain the Reactome, ChEBI, and HMDB resources used by `akg_omics.run_metabolomics_final`.

## 3. Configure the optional Agent

Rule mode does not call any external API. Agent mode uses an OpenAI-compatible chat-completions interface. In our final experiments, the Agent was configured with DeepSeek V4 Pro:

```bash
export KNOWLEDGE_AGENT_LLM_PROVIDER=deepseek
export KNOWLEDGE_AGENT_LLM_MODEL=deepseek-v4-pro
export KNOWLEDGE_AGENT_LLM_ENDPOINT=https://api.deepseek.com/v1/chat/completions
export DEEPSEEK_API_KEY=<your_deepseek_key>
```

You can also use another provider by setting:

```bash
export KNOWLEDGE_AGENT_LLM_PROVIDER=openai   # or openai_compatible, gemini, anthropic, qwen, glm
export KNOWLEDGE_AGENT_LLM_MODEL=<model_name>
export KNOWLEDGE_AGENT_LLM_ENDPOINT=<chat_completions_endpoint>
export KNOWLEDGE_AGENT_LLM_API_KEY=<your_api_key>
```

If no API key is provided, deterministic rule-based knowledge selection still works. The training scripts run both rule and agent variants where appropriate; the agent variant falls back safely when the LLM is disabled or unavailable.

## 4. Run proteogenomics tasks

Run the final 700-epoch, patience-30 setting for the four transcriptomic-proteomic tasks:

```bash
bash run_scripts/train_proteogenomics_rule_agent.sh
```

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash run_scripts/train_proteogenomics_rule_agent.sh
FINAL_SEEDS=1 bash run_scripts/train_proteogenomics_rule_agent.sh
PROTEO_DATA_ROOT=/path/to/spatch bash run_scripts/train_proteogenomics_rule_agent.sh
```

Additional ablation/configuration entrypoints:

```bash
bash run_scripts/train_proteogenomics_target_cycle_kb_align.sh
bash run_scripts/train_proteogenomics_dynamic_gate.sh
bash run_scripts/run_gene_panel_sweep.sh
```

## 5. Run metabolomics tasks

Run the final Task5-Task8 metabolomics configuration:

```bash
bash run_scripts/train_metabolomics_task5_8.sh
```

Baseline visualization is optional. To enable it, provide a compatible baseline repository:

```bash
RUN_BASELINE_VISUALIZATION=1 BASELINE_ROOT=/path/to/baseline_repo bash run_scripts/train_metabolomics_task5_8.sh
```

## 6. Visualize predictions

After training, generate spatial prediction figures:

```bash
bash run_scripts/visualize_predictions_only.sh
```

For direct re-export of Task1/Task2 arrays followed by figure generation:

```bash
bash run_scripts/visualize_all_tasks.sh
```

Key path overrides:

```bash
PROTEO_RUN_ROOT=/path/to/proteogenomics/run \
MET_ANALYSIS_ROOT=/path/to/metabolomics/detailed_analysis \
OUT_ROOT=/path/to/output \
bash run_scripts/visualize_predictions_only.sh
```

## Notes

- The default scripts are designed to reproduce the paper experiments on a Linux GPU server.
- Large datasets, KB files, caches, and trained results are intentionally not included.
- Do not commit API keys. Use environment variables such as `DEEPSEEK_API_KEY` or `KNOWLEDGE_AGENT_LLM_API_KEY`.
