from sklearn.neighbors import kneighbors_graph
from collections import Counter
from pandas import Series
from scipy import sparse
import numpy as np


def neighborhood_count_matrix(pos_data: np.ndarray, gene_ids: np.ndarray, k: int, include_self: bool = False):
    adj_mat = kneighbors_graph(pos_data, k)
    if include_self:
        adj_mat[np.diag_indices_from(adj_mat)] = 1.0

    nzi,nzj = adj_mat.nonzero();
    gene_counts = Series(gene_ids[nzj]).groupby(nzi).apply(list).map(Counter)

    js = np.concatenate(gene_counts.map(lambda x: list(x.keys())).values)
    inds = np.repeat(gene_counts.index, gene_counts.map(len)).values
    vals = np.concatenate(gene_counts.map(lambda x: list(x.values())).values)

    return sparse.csc_matrix((vals, (inds, js)))
