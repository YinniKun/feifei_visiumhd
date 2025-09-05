'''
Train a custom cell type annotation model using CellTypist and an atlas
'''

import celltypist
import scanpy as sc
import pandas as pd
import os
import argparse

def map_labels_to_adata(labels, adata):
    # Map the labels["cell_subtype__custom"] to the adata object
    # they are connected by the labels["cell_id"]
    adata.obs["cell_type"] = labels.set_index("cell_id").reindex(adata.obs.index)["cell_subtype__custom"]
    return adata


def main():
    parser = argparse.ArgumentParser(description="Train a custom CellTypist model using an atlas.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the atlas metadata file.")
    parser.add_argument("--seq_path", type=str, required=True, help="Path to the 10x data file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained model.")
    args = parser.parse_args()

    # read 10x
    adata_10x = sc.read_10x_mtx(args.seq_path, var_names='gene_symbols', cache=True)
    adata_10x.var_names_make_unique()
    # normalize
    sc.pp.normalize_total(adata_10x, target_sum=1e4)
    sc.pp.log1p(adata_10x)

    # read labels
    labels = pd.read_csv(args.labels_path)

    annotated_anndata = map_labels_to_adata(labels, adata_10x)

    print(f"10x data has {adata_10x.n_obs} observations and {adata_10x.n_vars} variables.")

    # Train the CellTypist model
    print("Training starting")
    model = celltypist.train(annotated_anndata, 'cell_type', n_jobs=10, max_iter=5, use_SGD=True)

    # Save the trained model
    model.write(os.path.join(args.output, 'model.pkl'))
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()