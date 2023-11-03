from typing import List, Optional
import numpy as np
import pandas as pd

def read_spatial_df(data_path:str, x_col:str="x", y_col:str="y", z_col:str="z", gene_col:str="gene", filter_cols:bool=False, min_molecules_per_gene:int=0):
    df_spatial = pd.read_csv(data_path)
    col_renames = {x_col: "x", y_col: "y", gene_col: "gene"}
    if (z_col is not None) and (z_col != "z"):
        if z_col not in df_spatial.columns:
            raise ValueError(f"z_col={z_col} is not in the data columns")
        col_renames[z_col] = "z"

    for (cn, co) in col_renames.items():
        if (cn != co) and (cn in df_spatial.columns):
            df_spatial.rename(columns={co: co + "_reserved"}, inplace=True)

    df_spatial.rename(columns=col_renames, inplace=True)
    if filter_cols:
        df_spatial = df_spatial[["x", "y", "z", "gene"]]

    if min_molecules_per_gene > 0:
        df_spatial = df_spatial[(df_spatial.gene.value_counts() >= min_molecules_per_gene)[df_spatial.gene].values]

    return df_spatial


# def load_df(data_path:str; min_molecules_per_gene:int=0, exclude_genes:list = None, **kwargs):
#     df_spatial = read_spatial_df(data_path; kwargs...)

#     gene_counts = StatsBase.countmap(df_spatial[!, :gene]);
#     large_genes = Set{String}(collect(keys(gene_counts))[collect(values(gene_counts)) .>= min_molecules_per_gene]);
#     df_spatial = df_spatial[in.(df_spatial.gene, Ref(large_genes)),:];

#     if length(exclude_genes) > 0:
#         exclude_genes = match_gene_names(exclude_genes, unique(df_spatial.gene))
#         df_spatial = df_spatial[.!in.(df_spatial.gene, Ref(exclude_genes)),:];
#         @info "Excluding genes: " * join(sort(collect(exclude_genes)), ", ")

#     df_spatial[!, :x] = Array{Float64, 1}(df_spatial[!, :x])
#     df_spatial[!, :y] = Array{Float64, 1}(df_spatial[!, :y])
#     df_spatial[!, :gene], gene_names = encode_genes(df_spatial[!, :gene]);
#     return df_spatial, gene_names;


def staining_value_per_point(pos_data: np.ndarray, img: np.ndarray, columns: Optional[List[str]]=None):
    """
    Extracts the staining value for each point in pos_data from the image img.

    Parameters
    ----------
    pos_data : np.ndarray
        Array of shape (N, 2) or (N, 3) containing the x, y, and z coordinates of each point.
    img : np.ndarray
        Image array of shape (W, H, C) or (W, H) where W is the width, H is the height, and C is the number of channels.
    columns : Optional[List[str]], optional
        List of channel names in the image, by default None. If provided, the output will be a pandas DataFrame with the columns named accordingly.
    """
    assert pos_data.shape[1] in (2, 3), "pos_data must have either 2 or 3 columns"
    if len(img.shape) == 3:
        if pos_data.shape[1] == 3:
            stain_vals = img[pos_data[:,2].astype(int), pos_data[:,1].astype(int), pos_data[:,0].astype(int)].T
        else:
            stain_vals = img[:, pos_data[:,1].astype(int), pos_data[:,0].astype(int)].T
            if columns is not None:
                stain_vals = pd.DataFrame(stain_vals, columns=columns)
    elif len(img.shape) == 2:
        stain_vals = img[pos_data[:,1].astype(int), pos_data[:,0].astype(int)].T
    else:
        raise ValueError("img must have either 2 or 3 dimensions")

    return stain_vals
