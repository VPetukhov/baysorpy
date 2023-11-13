cimport cython
cimport numpy as cnp
import os
from cython.parallel import prange
import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from libc.stdio cimport printf
import os
import numpy as np
from scipy.spatial import Delaunay
import sklearn
import scipy
from sklearn.metrics.pairwise import euclidean_distances



cdef class AdjList:
    cdef public list ids
    cdef public list weights

    def __init__(self, ids, weights):
        self.ids = [np.asarray(id_array, dtype=np.int64) for id_array in ids]
        self.weights = [np.asarray(weight_array, dtype=np.float64) for weight_array in weights]

def position_data(df):
    if 'z' in df.columns:
        return np.array(df[['x', 'y', 'z']])
    return np.array(df[['x', 'y']])

cpdef tuple _filter_long_edges(cnp.ndarray edge_list, cnp.ndarray adj_dists, float n_mads=2.0):
    cdef cnp.ndarray adj_dists_log = np.log10(adj_dists)
    cdef float d_threshold = np.median(adj_dists_log) + n_mads * np.median(np.abs(adj_dists_log - np.median(adj_dists_log)))

    cdef cnp.ndarray filt_mask = adj_dists_log < d_threshold
    return edge_list[:, filt_mask], adj_dists[filt_mask]


def normalize_points(cnp.ndarray[cnp.float64_t, ndim=2] points):
    points = points.copy()
    
    points -= np.min(points, axis=1).reshape(-1, 1)
    points /= np.max(points) * 1.1
    points += 1.01

    kdtree = scipy.spatial.KDTree(points.T)
    nn_dists, _ = kdtree.query(points.T, 2)
    is_duplicated = nn_dists[:, 1] < 1e-6
    num_dims = points.shape[0]
    random_shift = (np.random.rand(num_dims, is_duplicated.sum()) - 0.5) * 2e-5
    points[:, is_duplicated] += random_shift

    return points

cpdef tuple _adjacency_list(cnp.ndarray points, bint filter=False, float n_mads=2.0, int k_adj=5, str adjacency_type='auto', str distance='euclidean', bint return_tesselation=False):
    if adjacency_type == 'auto':
        adjacency_type = 'knn' if points.shape[0] == 3 else 'triangulation'
    if points.shape[0] == 3:
        if adjacency_type != 'knn':
            print("Warning: Only k-nn random field is supported for 3D data")
        adjacency_type = 'knn'
    else:
        assert points.shape[0] == 2, "Only 2D and 3D data is supported"
    
    points = normalize_points(points)
    cdef cnp.ndarray edge_list = np.empty((2, 0), dtype=int)
    if adjacency_type in ['triangulation', 'both']:
        tess = Delaunay(points.T, furthest_site=True)
        edge_list = np.array(tess.simplices).T
        if return_tesselation:
            return tess, points

    if adjacency_type in ['knn', 'both']:
        print(adjacency_type)
        kdtree = scipy.spatial.KDTree(points.T)
        distances, indices = kdtree.query(points.T, k=k_adj + 1)
        e_start = np.repeat(np.arange(len(indices)), k_adj)
        e_end = np.concatenate([i[1:] for i in indices])
        edges = np.vstack((np.minimum(e_start, e_end), np.maximum(e_start, e_end)))
        edge_list = np.unique(edges.T, axis=0).T


        # kdtree = KDTree(points.T)
        # _, indices = kdtree.query(points.T, k_adj + 1)
        # e_start = np.repeat(np.arange(len(indices)), k_adj)
        # e_end = indices[:, 1:].ravel() ### ?????
        # edges_knn = np.vstack((e_start, e_end))
        # edge_list = np.hstack((edges_knn, edge_list))

    cdef cnp.ndarray adj_dists = np.sqrt(np.sum((points[:, edge_list[0]] - points[:, edge_list[1]])**2, axis=0))

    if filter:
        edge_list, adj_dists = _filter_long_edges(edge_list, adj_dists, n_mads)

    return edge_list, adj_dists



def _cython_bincount(cnp.int64_t[:] arr):
    cdef int64_t max_value = arr[0]
    cdef int i, n = arr.shape[0]
    cdef int64_t *counts

    # Find the maximum value in the array
    for i in range(1, n):
        if arr[i] > max_value:
            max_value = arr[i]

    # Allocate memory for the counts (initialize to 0)
    counts = <int64_t *>malloc((max_value + 1) * sizeof(int64_t))
    if counts == NULL:
        raise MemoryError("Failed to allocate memory for counts array.")
    for i in range(max_value + 1):
        counts[i] = 0

    # Count occurrences
    for i in range(n):
        counts[arr[i]] += 1

    # Convert counts to a Python list (or a NumPy array if preferred)
    result = [counts[i] for i in range(max_value + 1)]

    # Free the allocated memory
    free(counts)

    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _split(cnp.ndarray array, cnp.ndarray[cnp.int64_t, ndim=1] factor, int max_factor=-1, bint drop_zero=False):
    cdef:
        Py_ssize_t i, fac, li
        list splitted = []
        #cnp.int64_t[:] counts
        cnp.int64_t[:] last_id
        int dtype_size = array.itemsize
        cnp.int64_t max_value = -2 

    assert array.shape[0] == factor.shape[0]
    
    # If max_factor isn't provided, compute max value of factor
    if max_factor == -1:
        for i in range(factor.shape[0]):
            if factor[i] > max_value and factor[i] != -1: 
                max_value = factor[i]
    else:
        max_value = max_factor
    
    counts = _cython_bincount(factor)
    
    splitted = [np.empty(c, dtype=array.dtype) for c in counts]
    last_id = np.zeros(max_value + 1, dtype=np.int64)

    for i in range(array.shape[0]):
        fac = factor[i]
        
        # Skip if drop_zero is True and factor is zero
        if drop_zero and fac == 0:
            continue

        # Ensure factor is within bounds
        assert 0 <= fac < last_id.shape[0], f"factor index out of bounds. {fac}"
        
        li = last_id[fac]
        
        # Ensure we are within bounds of the splitted list
        assert li < splitted[fac].shape[0], f"Index li out of bounds for splitted list. Error at {i}, li = {li}"

        splitted[fac][li] = array[i]
        last_id[fac] += 1

    return splitted

@cython.boundscheck(False)
@cython.wraparound(False)
def _convert_edge_list_to_adj_list(cnp.ndarray[cnp.int_t, ndim=2] edge_list, 
                                  cnp.ndarray[cnp.float64_t, ndim=1] edge_weights=None, 
                                  int n_verts=-1):

    # Infer the number of vertices if not provided
    if n_verts == -1:
        n_verts = edge_list.max()

    # Allocate memory for the result
    cdef cnp.ndarray res_ids
    cdef cnp.ndarray res_weights

    res_ids = np.concatenate((_split(edge_list[1], edge_list[0], n_verts),
                        _split(edge_list[0], edge_list[1], n_verts)))

    if edge_weights is None:
        return res_ids

    res_weights = np.concatenate((_split(edge_weights, edge_list[0], n_verts),
                        _split(edge_weights, edge_list[1], n_verts)))
    return res_ids, res_weights

cdef _build_molecule_graph(cnp.ndarray points, float min_edge_quant=0.3):
    edge_list, adjacent_dists = _adjacency_list(points) # ok
    
    min_edge_length = np.quantile(adjacent_dists, min_edge_quant)
    adjacent_weights = min_edge_length / np.maximum(adjacent_dists, min_edge_length)

    adjacent_points, adjacent_weights = _convert_edge_list_to_adj_list(edge_list, adjacent_weights, n_verts=points.shape[1])

    return AdjList(adjacent_points, adjacent_weights)

def build_molecule_graph(points):
    return _build_molecule_graph(points)