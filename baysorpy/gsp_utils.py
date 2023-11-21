import numpy as np
from scipy import sparse
from sklearn.neighbors import KDTree
from pygsp import graphs, filters

from functools import partial


def heat_filter(x, lmax: float, tau: float = 30.0):
    return (np.exp(-tau * np.abs(x / lmax)))


def build_spatial_graph_knn(pos_data: np.array, k: int):
    # TODO: should use sklearn
    # adj_mat = kneighbors_graph(pos_data, k, mode='distance', include_self=include_self, n_jobs=n_jobs)
    kd = KDTree(pos_data)
    adj_ids = [ids[ids != i] for i,ids in enumerate(kd.query(pos_data, k=(k+1))[1])]

    i = np.repeat(np.arange(len(adj_ids)), k)
    j = np.concatenate(adj_ids)
    v = np.ones(len(i))
    adj_mat = sparse.csc_matrix((v, (i, j)))
    adj_mat = (adj_mat + adj_mat.T) / 2
    return adj_mat


def get_heat_filter(graph: graphs.Graph, tau: float = 30.0):
    return filters.Filter(graph, partial(heat_filter, lmax=graph.lmax, tau=tau))


def smooth_signal(signal: np.array, graph: graphs.Graph, tau: float = 30.0):
    filt = get_heat_filter(graph, tau=tau)
    return filt.filter(signal, method="chebyshev", order=50)