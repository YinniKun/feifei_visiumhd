'''
Annotate cell types using clusterings (TooManyCells)
Then overlay marker genes on the clusters
'''
import anndata as ad
import toomanycells as tmc
import os
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Annotate cell types using clusterings (TooManyCells)")
    parser.add_argument("--input", type=str, required=True, help="Input AnnData file")
    parser.add_argument("--output", type=str, required=True, help="Output AnnData file")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    adata = tmc.annotate_cell_types(adata)
    adata.write_h5ad(args.output)

if __name__ == "__main__":
    main()



