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