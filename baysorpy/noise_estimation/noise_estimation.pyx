import numpy as np
import tqdm
cimport numpy as np
from scipy.stats import norm
from sklearn.neighbors import KDTree
from libc.math cimport exp, sqrt
# Typing declarations
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
from libc.math cimport exp, sqrt, M_PI
from libc.math cimport exp, log, sqrt, pow
from cython.parallel import prange
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


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

cpdef tuple _filter_long_edges(cnp.ndarray edge_list, cnp.ndarray adj_dists, float n_mads=2.0):
    cdef cnp.ndarray adj_dists_log = np.log10(adj_dists)
    cdef float d_threshold = np.median(adj_dists_log) + n_mads * np.median(np.abs(adj_dists_log - np.median(adj_dists_log)))

    cdef cnp.ndarray filt_mask = adj_dists_log < d_threshold
    return edge_list[:, filt_mask], adj_dists[filt_mask]


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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void wmean_std(cnp.float64_t[:] values, cnp.float64_t[:] weights, cnp.float64_t[:] result) nogil:
    cdef double w = 0.0, w_sq = 0.0, w_x = 0.0, w_x_sq = 0.0, m, s2

    for i in range(values.shape[0]):
        w += weights[i]
        w_sq += weights[i]**2
        w_x += weights[i]*values[i]
        w_x_sq += weights[i]*values[i]**2

    m = w_x / w
    s2 = (w_x_sq / w) - m**2
    result[0] = m
    result[1] = s2**0.5


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double] _norm_pdf(double[:] edge_lengths, double d1_mean, double d1_std):
    cdef int n = edge_lengths.shape[0]
    cdef np.ndarray[double] results = np.empty(n, dtype=np.float64)
    cdef double inv_sqrt_2pi = 1 / sqrt(2 * M_PI)
    cdef double std_inv = 1 / d1_std
    cdef double normalizing_factor = inv_sqrt_2pi * std_inv
    cdef double exponent_factor, value

    for i in range(n):
        exponent_factor = (edge_lengths[i] - d1_mean) * std_inv
        value = normalizing_factor * exp(-0.5 * exponent_factor * exponent_factor)
        results[i] = value

    return results

@cython.boundscheck(False)
@cython.wraparound(False)
cdef expect_noise_probabilities(
    double[:,:] assignment_probs,
    double d1_mean,
    double d2_mean,
    double d1_std,
    double d2_std,
    list ids,
    list weights,
    cnp.float64_t[:] edge_lengths,
    cnp.int64_t[:] updating_ids,
    double[:] min_confidence=None):

    cdef cnp.int64_t i, j, n1, n2
    cdef cnp.float64_t[:] norm_denses1, norm_denses2
    cdef double c_d1, c_d2, d1_val, d2_val, p1
    cdef cnp.int64_t[:] cur_points
    cdef cnp.float64_t[:] cur_weights
    cdef cnp.int64_t cur_points_len

    norm_denses1 = _norm_pdf(edge_lengths, d1_mean, d1_std)
    norm_denses2 = _norm_pdf(edge_lengths, d2_mean, d2_std)
    #print(np.array(norm_denses1))
    n1 = sum(assignment_probs[:, 0])
    n2 = len(assignment_probs) - n1

    for i in updating_ids:
        cur_points = ids[i]
        cur_weights = weights[i]
        cur_points_len = len(cur_points)
        
        c_d1 = c_d2 = 0.0

        for j in range(cur_points_len):
            ap = assignment_probs[cur_points[j], 0]
            c_d1 += cur_weights[j] * ap
            c_d2 += cur_weights[j] * (1. - ap)
        
        d1_val = n1 * exp(c_d1) * norm_denses1[i]
        d2_val = n2 * exp(c_d2) * norm_denses2[i]
        p1 = d1_val / max(d1_val + d2_val, 1e-20)
        if min_confidence is not None and min_confidence[i] > 0:
            p1 = min_confidence[i] + p1 * (1. - min_confidence[i])

        assignment_probs[i, 0] = p1
        assignment_probs[i, 1] = 1. - p1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void maximize_noise_distributions(cnp.float64_t[:] edge_lengths, cnp.float64_t[:,:] assignment_probs, cnp.float64_t[:, :] result):
    cdef:
        int i
        double[2] temp_result
        
    for i in range(2):
        wmean_std(edge_lengths, assignment_probs[:,i], temp_result)
        result[i, 0] = temp_result[0]
        result[i, 1] = temp_result[1]
 
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _get_assignment_probs(double[:] edge_lengths, double[:] min_confidence=None):
    # Initialization
    cdef double init_std, param_diff
    cdef np.int64_t i, n_iters, n1, n2
    cdef double[:,:] assignment_probs
    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] outlier_mask_np
    #cdef bint[:] outlier_mask_np
    cdef np.int64_t[:] updating_ids_np
    
    init_means = np.quantile(edge_lengths, [0.1, 0.9])
    init_std = (init_means[1] - init_means[0]) / 4.0
    d1, d2 = norm(loc=init_means[0], scale=init_std), norm(loc=init_means[1], scale=init_std)
    assignment_probs = np.column_stack((norm.pdf(edge_lengths, loc=d1.mean(), scale=d1.std()),
                                        norm.pdf(edge_lengths, loc=d2.mean(), scale=d2.std())))

    outlier_mask_np = np.array(edge_lengths > (init_means[1] + 3 * init_std), dtype=bool)
    updating_ids_list = [i for i, val in enumerate(outlier_mask_np) if not val]
    updating_ids_np = np.array(updating_ids_list, dtype=np.int64)    
    for i in range(len(outlier_mask_np)):
        if outlier_mask_np[i]:
            assignment_probs[i, 1] = 1.0
            
    assignment_probs /= np.sum(assignment_probs, axis=1, keepdims=True)
    return assignment_probs, d1, d2, edge_lengths, updating_ids_np, outlier_mask_np, min_confidence

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _fit_noise_probabilities(double[:] edge_lengths, adj_list, double[:] min_confidence=None, int max_iters=10000, double tol=0.005):
    cdef double init_std, param_diff
    cdef np.int64_t i, n_iters, n1, n2
    cdef cnp.float64_t[:,:] assignment_probs
    cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] outlier_mask_np
    cdef np.int64_t[:] updating_ids_np
    cdef cnp.float64_t[:, :] dists = np.empty((2, 2), dtype=np.float64)

    assignment_probs, d1, d2, edge_lengths, updating_ids_np, outlier_mask_np, min_confidence = _get_assignment_probs(edge_lengths, min_confidence)
    #print(np.array(assignment_probs))
    max_diffs = np.zeros(max_iters, dtype=np.float64)
    n_iters = max_iters
    for i in tqdm.tqdm(range(max_iters)):
        expect_noise_probabilities(assignment_probs, d1.mean(), d2.mean(), d1.std(), d2.std(), adj_list.ids, adj_list.weights, edge_lengths, updating_ids_np, min_confidence)
        #print(f'iter {i}', np.array(assignment_probs))
        maximize_noise_distributions(edge_lengths, assignment_probs, dists)
        d1n_m, d1n_s = dists[0][0], dists[0][1]
        d2n_m, d2n_s = dists[1][0], dists[1][1]
        # Estimate parameter differences as convergence criteria
        param_diff = max(
            abs(d1n_m - d1.mean()) / d1.mean(),
            abs(d2n_m - d2.mean()) / d2.mean(),
            abs(d1n_s - d1.std()) / d1.std(),
            abs(d2n_s - d2.std()) / d2.std()
        )
        max_diffs[i] = param_diff
        d1, d2 = norm(loc=d1n_m, scale=d1n_s), norm(loc=d2n_m, scale=d2n_s)
        #print(d1.mean(), d1.std(), d2.mean(), d2.std())
        if param_diff < tol:
            print(param_diff - tol)
            n_iters = i
            break

    print('Converged...')
    if d1.mean() > d2.mean():
        d1, d2 = d2, d1
    
    n1 = np.sum(assignment_probs[:, 0])
    n2 = len(assignment_probs) - n1

    cdef d1_mean =  d1.mean()
    cdef d1_std  =  d1.std()
    cdef d2_mean =  d2.mean()
    cdef d2_std  =  d2.std()
    cdef cnp.float64_t inv_sqrt_2pi = 1 / sqrt(2 * M_PI)
    cdef cnp.float64_t d1_std_inv = 1 / d1_std
    cdef cnp.float64_t d2_std_inv = 1 / d2_std
    cdef cnp.float64_t normalizing_factor_d1 = n1 * inv_sqrt_2pi * d1_std_inv
    cdef cnp.float64_t normalizing_factor_d2 = n2 * inv_sqrt_2pi * d2_std_inv
    cdef cnp.float64_t exponent_factor, pdf_value_d1, pdf_value_d2, _sum

    for i in range(len(edge_lengths)):
        exponent_factor = (edge_lengths[i] - d1_mean) * d1_std_inv
        pdf_value_d1 = normalizing_factor_d1 * exp(-0.5 * exponent_factor * exponent_factor)
    
        exponent_factor = (edge_lengths[i] - d2_mean) * d2_std_inv
        pdf_value_d2 = normalizing_factor_d2 * exp(-0.5 * exponent_factor * exponent_factor)
    
        if outlier_mask_np[i]:
            pdf_value_d2 = 1.0

        _sum = pdf_value_d1 + pdf_value_d2
        assignment_probs[i, 0] = pdf_value_d1 / _sum
        assignment_probs[i, 1] = pdf_value_d2 / _sum

    #if min_confidence is not None:
    #    for i in range(len(assignment_probs)):
    #        assignment_probs[i, 0] = min_confidence + assignment_probs[i, 0] * (1. - min_confidence)
    #        assignment_probs[i, 1] = 1.0 - assignment_probs[i, 0]

    cdef np.int64_t[:] assignment = np.empty_like(edge_lengths, dtype=np.int64)
    for i in range(len(assignment_probs)):
        if assignment_probs[i, 1] > 0.5:
            assignment[i] = 2
        else:
            assignment[i] = 1
    return assignment_probs, assignment, (d1, d2), max_diffs

def get_assignment_probs(edge_lengths, min_confidence=None):
    return _get_assignment_probs(edge_lengths, min_confidence)

def split(array, factor, max_value=-1, drop_zero=False):
    if max_value == -1:
        max_value = np.max(factor)
    return _split(array, factor, max_value, drop_zero)

def adjacency_list(points):
    return _adjacency_list(points)
def cython_bincount(values):
    return _cython_bincount(values)
def fit_noise_probabilities(edge_lengths, adj_list, min_confidence=None, max_iters=10000, tol=0.005):
    return _fit_noise_probabilities(edge_lengths, adj_list, min_confidence, max_iters, tol)
    
def estimate_confidence(df_spatial, nn_id, prior_assignment=None, prior_confidence=0.5):
    pos_data = position_data(df_spatial).T
    tree = KDTree(pos_data.T)
    dists, indices = tree.query(pos_data.T, k=nn_id + 1, sort_results=True)
    mean_dists = dists[:, nn_id]
   
    if prior_assignment is not None:
        min_confidence = prior_confidence ** 2 * (prior_assignment > 0)
    else:
        min_confidence = None

    adj_list = build_molecule_graph(pos_data)

    assignment_probs, assignment, (d1, d2), max_diffs = fit_noise_probabilities(mean_dists, adj_list)

    return mean_dists, assignment_probs[:, 0], min_confidence