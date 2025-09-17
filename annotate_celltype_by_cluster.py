'''
Annotate cell types using clusterings (TooManyCells)
Then overlay marker genes on the clusters
'''
import anndata as ad
from toomanycells import TooManyCells as tmc
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from qc+segment import plot_clusters_and_save_image
import scanpy as sc

def run_tmc_clustering(adata, tmc_output_dir):
    tmc_obj = tmc(adata, tmc_output_dir)
    tmc_obj.run_spectral_clustering()
    tmc_obj.store_outputs()
    return tmc_obj.A

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
        freq = adata.obs["sp_cluster"].value_counts(normalize=True)
        for cell_type in all_cell_types:
            proportions_df.loc[exp_name[i], cell_type] = freq.get(cell_type, 0)
    
    return proportions_df

def main():
    parser = argparse.ArgumentParser(description="Annotate cell types using clusterings (TooManyCells)")
    parser.add_argument('--adata', type=str, required=True, nargs="+", help='Path to the input segmented and binned anndata file (.h5ad)')
    parser.add_argument('--img', type=str, required=True, nargs="+", help='Path to the input histology image file (.tif)')
    parser.add_argument('--gdf', type=str, required=True, nargs="+", help='Path to the input binned cells geojson file (.geojson)')
    parser.add_argument('--exp_name', type=str, required=True, nargs="+", help='Experiment name for each Anndata file')
    parser.add_argument("--output", type=str, required=True, help="Output AnnData file and plots")
    args = parser.parse_args()
    
    plot_dir = os.path.join(args.output, "plots")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    adatas = []
    for i, file_path in enumerate(args.adata):
        adata = ad.read_h5ad(file_path)
        img = plt.imread(args.img[i])
        gdf = gpd.read_file(args.gdf[i])
        print(f"Loading {file_path} with {adata.n_obs} observations and {adata.n_vars} variables.")
        # # make gene symbols the var_names
        # if 'gene_symbols' in adata.var:
        #     adata.var['gene_ids'] = adata.var_names
        #     adata.var_names = adata.var['gene_symbols']
        #     #drop the gene_symbols column
        #     adata.var.drop(columns='gene_symbols', inplace=True)
        # adata.var_names = adata.var_names.astype(str)
        # adata.var_names_make_unique()
        #log1p transform the data
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # run tmc clustering to assign cell clusters
        # the cluster is stored at adata.obs['sp_cluster]
        new_adata = run_tmc_clustering(adata, os.path.join(args.output, "tmc_output"))
        adatas.append(new_adata)
        # save the clustered anndata
        adata.write_h5ad(file_path.replace('.h5ad', '_TMCannotated.h5ad'))
        print(f"Annotated anndata object saved as {file_path.replace('.h5ad', '_TMCannotated.h5ad')}.")
        # plot the clusters
        plot_clusters_and_save_image(
            title=f"Clusters for {args.exp_name[i]}",
            gdf=gdf,
            img=img,
            adata=new_adata,
            color_by_obs='sp_cluster',
            output_name=os.path.join(plot_dir, f"{args.exp_name[i]}_clusters.png")
        )

    # plot the cluster distributions
    prop_df = cell_type_proportions(adatas, all_cell_types, args.exp_name)
    prop_df.to_csv(os.path.join(plot_dir, "cell_type_proportions.csv"))
    print(f"Cell type proportions saved to {os.path.join(plot_dir, 'cell_type_proportions.csv')}.")


if __name__ == "__main__":
    main()



