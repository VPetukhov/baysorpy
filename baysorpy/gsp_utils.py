import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from pygsp import graphs, filters

from functools import partial


def heat_filter(x, lmax: float, tau: float = 30.0):
    return (np.exp(-tau * np.abs(x / lmax)))


def build_spatial_graph_knn(pos_data: np.array, k: int):
    adj_mat = kneighbors_graph(pos_data, k)
    return adj_mat + adj_mat.T


def get_heat_filter(graph: graphs.Graph, tau: float = 30.0):
    return filters.Filter(graph, partial(heat_filter, lmax=graph.lmax, tau=tau))


def smooth_signal(signal: np.array, graph: graphs.Graph, tau: float = 30.0):
    filt = get_heat_filter(graph, tau=tau)
    return filt.filter(signal, method="chebyshev", order=50)