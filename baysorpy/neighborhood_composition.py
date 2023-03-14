from sklearn.neighbors import kneighbors_graph
from collections import Counter
from pandas import Series
from scipy import sparse
import numpy as np

from .baysor_wrappers import neighborhood_count_matrix_jl


def neighborhood_count_matrix_py(
        pos_data: np.ndarray, gene_ids: np.ndarray, k: int,
        include_self: bool = False, n_genes: int = None, max_dist: float = -1.0, n_jobs: int = -1
    ):
    if n_genes is None:
        n_genes = np.max(gene_ids) + 1

    adj_mat = kneighbors_graph(pos_data, k, mode='distance', include_self=include_self, n_jobs=n_jobs)
    if max_dist > 0:
        adj_mat[adj_mat > max_dist] = 0

    nzi,nzj = adj_mat.nonzero();
    gene_counts = Series(gene_ids[nzj]).groupby(nzi).apply(list).map(Counter)

    js = np.concatenate(gene_counts.map(lambda x: list(x.keys())).values)
    inds = np.repeat(gene_counts.index, gene_counts.map(len)).values
    vals = np.concatenate(gene_counts.map(lambda x: list(x.values())).values)

    return sparse.csc_matrix((vals, (inds, js)), shape=(pos_data.shape[0], n_genes))


def neighborhood_count_matrix(pos_data: np.ndarray, gene_ids: np.ndarray, k: int, method='py', **kwargs):
    """
    Compute the neighborhood count matrix for a given set of spatial coordinates and gene ids.
    Args:
        pos_data: Spatial coordinates of cells (2-3D numpy array of shape (n_molecules, n_dims))
        gene_ids: Gene ids of cells (1D numpy array of shape (n_molecules,))
        k: Number of nearest neighbors
        method: python ('py') or Julia ('jl') implementation. The later is ~5 times faster, but requires JuliaCall.
    """
    if method == 'py':
        return neighborhood_count_matrix_py(pos_data, gene_ids, k, **kwargs)

    if method == 'jl':
        return neighborhood_count_matrix_jl(pos_data, gene_ids, k, **kwargs)

    raise ValueError(f"Method {method} not found. Please choose from ['py', 'jl']")
