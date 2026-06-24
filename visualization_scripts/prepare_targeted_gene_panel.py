#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers", required=True)
    parser.add_argument("--original-order", required=True)
    parser.add_argument("--raw-gene-h5ad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()

    marker_data = json.loads(Path(args.markers).read_text(encoding="utf-8"))
    required = list(marker_data["protein_to_gene"].values())
    for genes in marker_data["coad"].values():
        required.extend(genes)
    marker_keys = {str(name).upper() for name in required}

    original = pd.read_csv(args.original_order)
    source_col = next(
        (name for name in ["feature_name", "gene", "gene_name", "symbol"] if name in original),
        original.columns[0],
    )
    required.extend(original[source_col].astype(str).tolist())

    adata = sc.read_h5ad(args.raw_gene_h5ad, backed="r")
    available = {str(name).upper(): str(name) for name in adata.var_names}
    adata.file.close()

    ordered = []
    seen = set()
    for requested in required:
        key = str(requested).upper()
        actual = available.get(key)
        if actual is None or actual in seen:
            continue
        ordered.append(actual)
        seen.add(actual)
        if len(ordered) >= args.top_k:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature_name": ordered,
            "priority_rank": range(1, len(ordered) + 1),
            "required_marker": [
                str(name).upper() in marker_keys for name in ordered
            ],
        }
    ).to_csv(output, index=False)
    print(f">>> selected={len(ordered)}")
    print(f">>> output={output}")


if __name__ == "__main__":
    main()
