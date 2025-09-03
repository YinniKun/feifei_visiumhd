'''
This module is used to annotate cell types once the cells have been segmented and binned from VisiumHD
It reads an anndata file and adds a new layer for cell type annotation using CellTypist.
'''

import pandas as pd
import anndata as ad
import celltypist
import os
import matplotlib.pyplot as plt
from celltypist import models, annotate
import scanpy as sc
import distinctipy
from matplotlib.colors import ListedColormap
import numpy as np
import argparse

def annotate_cell_types(adata: ad.AnnData, model_name: str = "Adult_Human_Skin.pkl") -> ad.AnnData:
    '''Annotate cell types in the anndata object using a pre-trained CellTypist model.

    Args:
        adata (AnnData): The anndata object containing the cell data.
        model_name (str): Name of the pre-trained CellTypist model.
    
    Returns:
        ad.AnnData: The anndata object with an additional 'cell_type' column in obs.
    '''

    model = models.Model.load(model=model_name)
    # convert to mouse
    if "human" in model_name.lower():
        model.convert()

    # get all possible cell types by the model
    cell_types = model.cell_types

    #annotate
    annotations = annotate(adata, model)

    adata.obs['cell_type'] = annotations.predicted_labels

    return adata, cell_types

def cell_type_proportions(adatas: list[ad.AnnData], all_cell_types: list[str],
                          exp_name: list[str]) -> pd.DataFrame:
    '''Analyze and plot cell type proportions.

    Args:
        adatas (list[ad.AnnData]): List of anndata objects containing the cell type annotations.
        all_cell_types (list[str]): List of all possible cell types.
        exp_name (list[str]): List of experiment names corresponding to each anndata object.

    Returns:
        pd.DataFrame: Cell type proportions in the dataset.
    '''
    # build a proportions dataframe
    proportions_df = pd.DataFrame(index=exp_name, columns=all_cell_types)
    for i, adata in enumerate(adatas):
        # Count the occurrences of each cell type
        freq = adata.obs["cell_type"].value_counts(normalize=True)
        for cell_type in all_cell_types:
            proportions_df.loc[exp_name[i], cell_type] = freq.get(cell_type, 0)
    
    return proportions_df
        

def main():
    parser = argparse.ArgumentParser(description="Annotate cell types based on segmented and binned data.")
    parser.add_argument('--input', type=str, required=True, nargs="+", help='Path to the input segmented and binned anndata file (.h5ad)')
    parser.add_argument('--exp_name', type=str, required=True, nargs="+", help='Experiment name for each Anndata file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the cell type proportions plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Load the anndata objects
    adatas = []
    for file_path in args.input:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        adata = ad.read_h5ad(file_path)
        print(f"Loading {file_path} with {adata.n_obs} observations and {adata.n_vars} variables.")
        # make gene symbols the var_names
        if 'gene_symbols' in adata.var:
            adata.var['gene_ids'] = adata.var_names
            adata.var_names = adata.var['gene_symbols']
            #drop the gene_symbols column
            adata.var.drop(columns='gene_symbols', inplace=True)
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()
        #log1p transform the data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # print(f"After filtering: {adata.n_obs} observations and {adata.n_vars} variables.")
        
        # Annotate cell types
        adata, cell_types = annotate_cell_types(adata)

        # Save the annotated anndata object
        adata.write_h5ad(file_path.replace('.h5ad', '_annotated.h5ad'))
        print(f"Annotated anndata object saved as {file_path.replace('.h5ad', '_annotated.h5ad')}.")

        adatas.append(adata)
    
    cell_type_df = cell_type_proportions(adatas, cell_types, args.exp_name)
    cell_type_df.to_csv(os.path.join(args.output, "cell_type_proportions.csv"))

    # Plot cell type proportions
    # get a colour palette according to the number of cell types
    colors = distinctipy.get_colors(len(cell_types))
    cmap = ListedColormap(colors, name="maxdist40")
    # drop column with differentiated kc since there are too many
    if 'Differentiated_KC' in cell_type_df.columns:
        cell_type_df = cell_type_df.drop(columns='Differentiated_KC')
    cell_type_df.plot(kind="bar", figsize=(20,20), colormap=cmap)
    plt.ylabel("Relative Frequency")
    plt.xlabel("Experiment")
    plt.title("Relative Frequency of Cell Types per Experiment")
    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "cell_type_proportions_no_KC.png"))
    plt.close()

if __name__ == "__main__":
    main()