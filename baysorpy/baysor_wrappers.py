import numpy as np
from pandas import DataFrame
from typing import List
from scipy import sparse
# We don't import JuliaCall here as it requires LD_LIBRARY_PATH and can crash the session

def install_baysor(rev="develop", force=False):
    from juliacall import Main as jl
    jl.seval('using Pkg')
    is_installed = jl.seval('Base.UUID("cc9f9468-1fbe-11e9-0acf-e9460511877c") in keys(Pkg.dependencies())')
    if is_installed and not force:
        print("Baysor is already installed.")
        return False

    print("Installing Baysor...")
    jl.seval(f'Pkg.add(Pkg.PackageSpec(url="https://github.com/kharchenkolab/Baysor.git", rev="{rev}"))')
    return True


def load_module(module: str):
    from juliacall import Main as jl
    mod_aliases = {
        "DataLoading": "DAT",
        "Reporting": "REP",
        "Processing": "BPR",
        "CLI": "CLI"
    }

    if module not in mod_aliases:
        raise ValueError(f"Module {module} not found. Please choose from {mod_aliases.keys()}")

    module = mod_aliases[module]
    return jl.seval(f"import Baysor; Baysor.load_module(Baysor.{module})")


def estimate_confidence(df_spatial:DataFrame, nn_id:int, prior_assignment:List[int]=None, prior_confidence:float=0.5):
    from juliacall import Main as jl
    bpr = load_module("Processing")
    jl.seval("using DataFrames")
    conf = bpr.estimate_confidence(
        jl.DataFrame(df_spatial, copycols=False), prior_assignment, nn_id=nn_id, prior_confidence=prior_confidence
    )[1]

    return np.array(conf)


def neighborhood_count_matrix_jl(
        pos_data: np.ndarray, gene_ids: np.ndarray, k: int, normalize: bool = False, n_genes: int = None, **kwargs
    ):
    from juliacall import Main as jl
    bpr = load_module("Processing")
    jl.seval("using SparseArrays;")

    if n_genes is None:
        n_genes = np.max(gene_ids) + 1

    neighb_mat = bpr.neighborhood_count_matrix(
        jl.Matrix(pos_data.T), jl.Vector(gene_ids + 1), k, normalize=normalize,
        n_genes=n_genes, **kwargs
    )

    i,j,z = [np.array(x) for x in jl.findnz(neighb_mat)]
    neighb_mat = sparse.csc_matrix((z, (j - 1, i - 1)), shape=(pos_data.shape[0], n_genes))
    return neighb_mat
