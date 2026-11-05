# AKG-Omics

AKG-Omics integrates histology, molecular measurements, and biological knowledge for spatial cross-omics prediction.

## Installation

```bash
conda create -n akg-omics python=3.11 -y
conda activate akg-omics
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is required for GPU training.

## Data preparation

Place the section-level arrays in `data/<dataset>/<section>/`:

```text
data/
├── lymphnode/{A1,D1}/
├── tonsil/{A1,D1}/
├── mouse/{B1,C1}/
└── coad/{gene,protein}/
```

Each section directory contains the following NumPy arrays. Rows must follow the same spot order.

| File | Shape | Contents |
|---|---|---|
| `gene.npy` | spots × genes | Per-spot normalized RNA expression |
| `other.npy` | spots × features | Per-spot normalized protein or metabolite measurements |
| `he.npy` | spots × embedding dimensions | H&E features from a frozen image encoder |
| `coords.npy` | spots × 2 | Spatial coordinates |
| `ids.npy` | spots | Unique spot identifiers |
| `gene_names.npy` | genes | Unique gene symbols |
| `other_names.npy` | features | Unique protein names or metabolite feature identifiers |

For mouse data, also provide `gene_ids.npy` containing Ensembl gene IDs in the same order as `gene_names.npy`. An optional `other_symbols.npy` maps protein panel labels to gene symbols. Save identifiers as NumPy string arrays.

Use arrays before feature selection, fitted scaling, or PCA. The preparation step selects features and fits scaling/PCA on the source section, then applies the frozen transforms to permitted target inputs. Input locations are configured in `configs/<dataset>_inputs.json`.

Place knowledge resources in `assets/human/` or `assets/metabolite/`; filenames are listed in [assets/README.md](assets/README.md).

## Run

Prepare one task:

```bash
python run.py prepare --dataset tonsil --task 4 --output prepared/tonsil_T04
```

Train and evaluate using the prepared data:

```bash
python run.py run --dataset tonsil --task 4 --policy rule \
  --data prepared/tonsil_T04 --output outputs/tonsil_T04_rule
```

Alternatively, prepare data and run the model in one command:

```bash
python run.py run --dataset tonsil --task 4 --policy rule \
  --output outputs/tonsil_T04_rule
```

Replace `--dataset` with `coad`, `lymphnode`, `tonsil`, or `mouse`, and choose the corresponding task:

| Task | Input | Prediction | Datasets |
|---|---|---|---|
| T1 | H&E | RNA | coad, lymphnode, tonsil |
| T2 | H&E | Protein | coad, lymphnode, tonsil |
| T3 | H&E + protein | RNA | coad, lymphnode, tonsil |
| T4 | H&E + RNA | Protein | coad, lymphnode, tonsil |
| T5 | H&E | Metabolite features | mouse |
| T6 | H&E | RNA | mouse |
| T7 | H&E + RNA | Metabolite features | mouse |
| T8 | H&E + metabolite features | RNA | mouse |

Use `--policy agent` for the provided Agent configuration. Both policies use the source-selection settings in `configs/knowledge.json` by default. For fresh Agent selection, add `--knowledge select` and set `KNOWLEDGE_AGENT_LLM_PROVIDER`, `KNOWLEDGE_AGENT_LLM_MODEL`, and `KNOWLEDGE_AGENT_LLM_API_KEY` in the environment.

Use `--seed 1` to set the training seed and `--device cuda:0` to select a GPU (`--device cpu` for CPU execution). Results are written to `<output>/run/metrics.json` and `<output>/run/feature_metrics.npz`.

## Code structure

```text
akg_omics/
├── core/                 # Model architecture and training protocol
├── preprocessing.py      # Source-fitted feature selection, scaling, and PCA
├── data.py               # Task data loading
├── validation.py         # Input and fitted-transform validation
├── knowledge.py          # Knowledge construction
├── knowledge_agent/      # Knowledge-source selection
├── human_builder.py      # Gene/protein knowledge relations
├── metabolic_builder.py  # Gene/metabolite knowledge relations
├── trainer.py            # Training and evaluation
└── cli.py                # Command-line interface
```
