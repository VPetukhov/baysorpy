from noise_estimation cimport AdjList

from libc.math cimport log, log10, exp, M_PI, fabs
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
import copy

from cython.parallel import parallel, prange
import tqdm

cimport cython

cdef inline cnp.float64_t max_abs_diff(cnp.float64_t[:] col1, cnp.float64_t[:] col2):
    cdef int i, n = col1.shape[0]
    cdef cnp.float64_t max_diff = 0.0, diff

    for i in range(n):
        diff = fabs(col1[i] - col2[i])
        if diff > max_diff:
            max_diff = diff

    return max_diff

cdef estimate_difference_l0(cnp.float64_t[:, :] m1,
                             cnp.float64_t[:, :] m2,
                             cnp.float64_t[:] col_weights,
                             cnp.float64_t change_threshold=1e-7):
    if m1.shape[0] != m2.shape[0] or m1.shape[1] != m2.shape[1]:
        raise ValueError("Matrices must be of the same size")

    cdef cnp.float64_t max_diff = 0.0
    cdef int n_changed = 0
    cdef cnp.float64_t c_max
    cdef int ci, i, n_cols = m1.shape[1]

    for ci in range(n_cols):
        c_max = max_abs_diff(m1[:, ci], m2[:, ci])

        if col_weights is not None:
            c_max *= col_weights[ci]

        if c_max > change_threshold:
            n_changed += 1

        if c_max > max_diff:
            max_diff = c_max

    return max_diff, n_changed / n_cols

ctypedef struct _FNormal:
    double mu, sigma, c, s

cdef _FNormal createFNormal(double mu, double sigma) nogil:
    cdef double c, s
    cdef _FNormal _fNormal
    c = -0.5 * log(2 * M_PI) - log(sigma)
    s = 0.5 / sigma**2
    #_fNormal = <_FNormal *>malloc(sizeof(_FNormal))
    _fNormal.mu = mu
    _fNormal.sigma = sigma
    _fNormal.c = c
    _fNormal.s = s
    return _fNormal

cdef _NormalComponent createEmptyNormalComponent(int size) nogil:
    cdef _NormalComponent _normalComponent
    _normalComponent.size = size
    _normalComponent.n = 0
    _normalComponent.dists = <_FNormal *>malloc(_normalComponent.size * sizeof(_FNormal))

    return _normalComponent

cdef _NormalComponentList* createEmptyNormalComponentList(int size) nogil:
    cdef _NormalComponentList* _normalComponentList = <_NormalComponentList *>malloc(sizeof(_NormalComponentList))
    _normalComponentList.size = size
    _normalComponentList.data = <_NormalComponent *>malloc(_normalComponentList.size * sizeof(_NormalComponent))

    return _normalComponentList

cdef  int updateFNormal(double mu, double sigma, _FNormal* dest) nogil:
    cdef double c, s
    c = -0.5 * log(2 * M_PI) - log(sigma)
    s = 0.5 / sigma**2
    dest.mu = mu
    dest.sigma = sigma
    dest.c = c
    dest.s = s

    return 0

ctypedef struct _NormalComponent:
    int size
    _FNormal *dists
    double n

ctypedef struct _NormalComponentList:
    int size
    _NormalComponent *data

cdef class NormalComponentListWrapper:
    cdef _NormalComponentList* _c_list

    def __cinit__(self):
        self._c_list = NULL

    def __dealloc__(self):
        if self._c_list is not NULL:
            # Free the data if necessary, e.g., if it's dynamically allocated
            free(self._c_list.data)
            free(self._c_list)
    def get_some_data(self):
        if self._c_list is NULL:
            return None
        # Assuming your C struct has an attribute 'some_data' that you want to access
        return self._c_list.some_data

    # Example method to manipulate C struct data
    def set_some_data(self, value):
        if self._c_list is not NULL:
            # Perform necessary type checking and conversion
            self._c_list.some_data = value

cdef double normal_logpdf(_FNormal n, double v) nogil:
    return n.c - (v - n.mu)**2 * n.s

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double pdf(_NormalComponent comp, double[:] vec) nogil:
    cdef double dens = 0.0
    cdef int i
    for i in range(vec.shape[0]):
        dens += normal_logpdf(comp.dists[i], vec[i])

    return comp.n * exp(dens)

cdef double comp_pdf(_NormalComponentList* cell_type_exprs, int ci, double[:] gene_or_factor) nogil:
    return pdf(cell_type_exprs.data[ci], gene_or_factor)

# cpdef double get_gene_vec(genes, int i):
#     if isinstance(genes, cnp.ndarray):
#         if genes.ndim == 1:  # Vector
#             return genes[i]
#         elif genes.ndim == 2:  # Matrix
#             return genes[i, :]
#         else:
#             raise ValueError("Invalid dimension for genes")
#     else:
#         raise ValueError("Invalid type for genes")

cdef cnp.int64_t *get_int_array_lengths(py_array):

    # Step 1: Declare the C array type
    cdef cnp.int64_t *c_array
    cdef int i

    # Step 2: Allocate memory for the C array
    c_array = <cnp.int64_t *>malloc(len(py_array) * sizeof(cnp.int64_t))
    for i in range(len(py_array)):
        # Assuming each element in py_array is a cnp.ndarray
        sub_array = py_array[i]
        c_array[i] = len(sub_array)

    return c_array

cdef cnp.int64_t **convert_int_array(py_array):
    # Step 1: Declare the C array type
    cdef cnp.int64_t **c_array
    cdef int i, j

    # Step 2: Allocate memory for the C array
    c_array = <cnp.int64_t **>malloc(len(py_array) * sizeof(cnp.int64_t *))
    for i in range(len(py_array)):
        # Assuming each element in py_array is a cnp.ndarray
        sub_array = py_array[i]
        c_array[i] = <cnp.int64_t *>malloc(len(sub_array) * sizeof(cnp.int64_t))

        # Step 3: Copy the data with type checking
        for j in range(len(sub_array)):
            # Here you may need to cast or convert the element to cnp.int64_t
            c_array[i][j] = <cnp.int64_t>sub_array[j]


    return c_array

cdef cnp.float64_t **convert_float_array(py_array):

    # Step 1: Declare the C array type
    cdef cnp.float64_t **c_array
    cdef int i, j

    # Step 2: Allocate memory for the C array
    c_array = <cnp.float64_t **>malloc(len(py_array) * sizeof(cnp.float64_t *))
    for i in range(len(py_array)):
        # Assuming each element in py_array is a cnp.ndarray
        sub_array = py_array[i]
        c_array[i] = <cnp.float64_t *>malloc(len(sub_array) * sizeof(cnp.float64_t))

        # Step 3: Copy the data with type checking
        for j in range(len(sub_array)):
            # Here you may need to cast or convert the element to cnp.float64_t
            c_array[i][j] = <cnp.float64_t>sub_array[j]


    return c_array

cpdef cluster_molecules_loop(
        cnp.float64_t[:, :] assignment_probs, 
        cnp.float64_t[:, :] assignment_probs_prev, 
        cnp.float64_t[:, :] cell_type_exprs,
        cnp.float64_t[:] confidence,
        cnp.int64_t[:] genes,
        AdjList adj_list,
        cnp.int64_t max_iters,
        cnp.float64_t tol,
        cnp.int64_t n_iters_without_update,
        bint do_maximize,
        cnp.float64_t mrf_weight):

    cdef cnp.ndarray[cnp.float64_t, ndim=1] max_diffs = np.zeros(max_iters, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] change_fracs = np.zeros(max_iters, dtype=np.float64)

    cdef cnp.int64_t ** adj_list_ids = convert_int_array(adj_list.ids)
    cdef cnp.int64_t * adj_list_ids_lengths = get_int_array_lengths(adj_list.ids)
    cdef cnp.float64_t ** adj_list_weights = convert_float_array(adj_list.weights)
    cdef cnp.int64_t adj_list_len = len(adj_list.ids)

    for i in tqdm.tqdm(range(1, max_iters + 1)):
        assignment_probs_prev = assignment_probs.copy()
        expect_molecule_clusters_optimized(
            assignment_probs, assignment_probs_prev, cell_type_exprs, genes, adj_list_ids, adj_list_ids_lengths, adj_list_weights, adj_list_len, mrf_weight=mrf_weight
        )

        if do_maximize:
            maximize_molecule_clusters(cell_type_exprs, genes, confidence, assignment_probs, add_pseudocount=True)

        md, cf = estimate_difference_l0(assignment_probs, assignment_probs_prev, confidence)
        max_diffs = np.append(max_diffs, md)
        change_fracs = np.append(change_fracs, cf)

        if i > n_iters_without_update and np.max(max_diffs[-n_iters_without_update:]) < tol:
            break

    if do_maximize:
        maximize_molecule_clusters(cell_type_exprs, genes, confidence, assignment_probs, add_pseudocount=False)

    free(adj_list_ids_lengths)

    for i in range(adj_list_len):
        free(adj_list_weights[i])
        free(adj_list_ids[i])

    free(adj_list_weights)
    free(adj_list_ids)

    return (max_diffs, change_fracs)

cpdef cluster_molecules_loop_norm(
        cnp.float64_t[:, :] assignment_probs, 
        cnp.float64_t[:, :] assignment_probs_prev, 
        NormalComponentListWrapper cell_type_exprs,
        cnp.float64_t[:] confidence,
        cnp.float64_t[:, :] genes,
        AdjList adj_list,
        cnp.int64_t max_iters,
        cnp.float64_t tol,
        cnp.int64_t n_iters_without_update,
        bint do_maximize,
        cnp.float64_t mrf_weight):

    cdef cnp.ndarray[cnp.float64_t, ndim=1] max_diffs = np.zeros(max_iters, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] change_fracs = np.zeros(max_iters, dtype=np.float64)

    cdef _NormalComponentList * components = cell_type_exprs._c_list

    cdef cnp.int64_t ** adj_list_ids = convert_int_array(adj_list.ids)
    cdef cnp.int64_t * adj_list_ids_lengths = get_int_array_lengths(adj_list.ids)
    cdef cnp.float64_t ** adj_list_weights = convert_float_array(adj_list.weights)
    cdef cnp.int64_t adj_list_len = len(adj_list.ids)

    for i in tqdm.tqdm(range(1, max_iters + 1)):
        assignment_probs_prev = assignment_probs.copy()
        expect_molecule_clusters_norm_optimized(
            assignment_probs, assignment_probs_prev, components, genes, adj_list_ids, adj_list_ids_lengths, adj_list_weights, adj_list_len, mrf_weight=mrf_weight
        )

        if do_maximize:
            maximize_molecule_clusters_norm_optimized(components, genes, confidence, assignment_probs, add_pseudocount=True)

        md, cf = estimate_difference_l0(assignment_probs, assignment_probs_prev, confidence)
        max_diffs = np.append(max_diffs, md)
        change_fracs = np.append(change_fracs, cf)

        if i > n_iters_without_update and np.max(max_diffs[-n_iters_without_update:]) < tol:
            break

    if do_maximize:
        maximize_molecule_clusters_norm_optimized(components, genes, confidence, assignment_probs, add_pseudocount=False)

    free(adj_list_ids_lengths)

    for i in range(adj_list_len):
        free(adj_list_weights[i])
        free(adj_list_ids[i])

    free(adj_list_weights)
    free(adj_list_ids)

    return (max_diffs, change_fracs)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double expect_molecule_clusters_optimized(
        cnp.float64_t[:, :] assignment_probs, 
        cnp.float64_t[:, :] assignment_probs_prev, 
        cnp.float64_t[:, :] cell_type_exprs,
        cnp.int64_t[:] genes,
        cnp.int64_t ** adj_list_ids,
        cnp.int64_t * adj_list_ids_lengths,
        cnp.float64_t ** adj_list_weights,
        cnp.int64_t adj_list_len,
        double mrf_weight=1.0, 
        bint only_mrf=False,
        cnp.float64_t[:] is_fixed=None):
    cdef int i, ri, j, k, l
    cdef double mrf_prior, a_p
    cdef cnp.float64_t * dense_sums
    cdef cnp.float64_t * c_ds
    cdef int gene

    dense_sums = <cnp.float64_t *>malloc(adj_list_len * sizeof(cnp.float64_t))
    c_ds = <cnp.float64_t *>malloc(adj_list_len * sizeof(cnp.float64_t))

    with nogil, parallel():
        for i in prange(adj_list_len):
            dense_sums[i] = 0.0
            # Handle is_fixed condition if necessary
            # if is_fixed is not None and not is_fixed[i]:
            #     continue

            gene = genes[i]
            for ri in range(assignment_probs.shape[0]):
                c_ds[i] = 0.0
                for j in range(adj_list_ids_lengths[i]):
                    a_p = assignment_probs_prev[ri, adj_list_ids[i][j]]
                    if a_p > 1e-5:
                        c_ds[i] += adj_list_weights[i][j] * a_p
                c_d_x_mfr_weight: cnp.float64_t = c_ds[i] * mrf_weight
                
                mrf_prior = exp(c_d_x_mfr_weight)
                if only_mrf:
                    assignment_probs[ri, i] = mrf_prior
                else:
                    assignment_probs[ri, i] = cell_type_exprs[ri, gene] * mrf_prior
                dense_sums[i] += assignment_probs[ri, i]
            if dense_sums[i] > 1e-20:
                for k in range(assignment_probs.shape[0]):
                    assignment_probs[k, i] = assignment_probs[k, i] / dense_sums[i]
            else:
                for k in range(assignment_probs.shape[0]):
                    assignment_probs[k, i] = 1 / assignment_probs.shape[0]

    free(dense_sums)
    free(c_ds)

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double expect_molecule_clusters_norm_optimized(
        cnp.float64_t[:, :] assignment_probs, 
        cnp.float64_t[:, :] assignment_probs_prev, 
        _NormalComponentList* cell_type_exprs,
        cnp.float64_t[:, :] genes,
        cnp.int64_t ** adj_list_ids,
        cnp.int64_t * adj_list_ids_lengths,
        cnp.float64_t ** adj_list_weights,
        cnp.int64_t adj_list_len,
        double mrf_weight=1.0, 
        bint only_mrf=False,
        cnp.float64_t[:] is_fixed=None):
    cdef int i, ri, j, k, l
    cdef double mrf_prior, a_p
    cdef cnp.float64_t * dense_sums
    cdef cnp.float64_t * c_ds

    dense_sums = <cnp.float64_t *>malloc(adj_list_len * sizeof(cnp.float64_t))
    c_ds = <cnp.float64_t *>malloc(adj_list_len * sizeof(cnp.float64_t))

    with nogil, parallel():
        for i in prange(adj_list_len):
            dense_sums[i] = 0.0
            # Handle is_fixed condition if necessary
            # if is_fixed is not None and not is_fixed[i]:
            #     continue

            for ri in range(assignment_probs.shape[0]):
                c_ds[i] = 0.0
                for j in range(adj_list_ids_lengths[i]):
                    a_p = assignment_probs_prev[ri, adj_list_ids[i][j]]
                    if a_p > 1e-5:
                        c_ds[i] += adj_list_weights[i][j] * a_p
                c_d_x_mfr_weight: cnp.float64_t = c_ds[i] * mrf_weight
                
                mrf_prior = exp(c_d_x_mfr_weight)
                if only_mrf:
                    assignment_probs[ri, i] = mrf_prior
                else:
                    assignment_probs[ri, i] = comp_pdf(cell_type_exprs, ri, genes[i]) * mrf_prior
                dense_sums[i] += assignment_probs[ri, i]
            if dense_sums[i] > 1e-20:
                for k in range(assignment_probs.shape[0]):
                    assignment_probs[k, i] = assignment_probs[k, i] / dense_sums[i]
            else:
                for k in range(assignment_probs.shape[0]):
                    assignment_probs[k, i] = 1 / assignment_probs.shape[0]

    free(dense_sums)
    free(c_ds)

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef maximize_molecule_clusters(cnp.float64_t[:, :] cell_type_exprs, cnp.int64_t[:] genes, 
                                 cnp.float64_t[:] confidence, cnp.float64_t[:, :] assignment_probs, 
                                 cnp.float64_t[:, :] prior_exprs=None, cnp.float64_t[:, :] prior_stds=None, 
                                 bint add_pseudocount=False):
    cdef int i, j, num_genes, num_cell_types
    cdef double t_conf
    local_sum_ptr = <double *>malloc(cell_type_exprs.shape[0] * sizeof(double))

    num_genes = genes.shape[0]
    num_cell_types = cell_type_exprs.shape[0]

    for i in range(cell_type_exprs.shape[0]):
        for j in range(cell_type_exprs.shape[1]):
            cell_type_exprs[i][j] = 0.0
    
    for i in range(num_genes):
        t_gene = genes[i]
        t_conf = confidence[i]

        for j in range(num_cell_types):
            cell_type_exprs[j, t_gene] += t_conf * assignment_probs[j, i]

    #if prior_exprs is not None:
    #    cdef cnp.ndarray[double, ndim=2] mult = np.sum(cell_type_exprs, axis=1, keepdims=True)
    #    for i in range(num_cell_types):
    #        for j in range(cell_type_exprs.shape[1]):
    #            cell_type_exprs[i, j] = adj_value_norm(cell_type_exprs[i, j], 
    #                                                   prior_exprs[i, j] * mult[i, 0], 
    #                                                   prior_stds[i, j] * mult[i, 0])

    # calculate local sum
    for i in range(cell_type_exprs.shape[0]):
        local_sum_ptr[i] = 0.0
        for j in range(cell_type_exprs.shape[1]):
            local_sum_ptr[i] += cell_type_exprs[i][j]

    with nogil, parallel():
        if add_pseudocount:
            for j in prange(cell_type_exprs.shape[1]):
                for i in range(cell_type_exprs.shape[0]):
                    cell_type_exprs[i][j] = cell_type_exprs[i][j] + 1
                    cell_type_exprs[i][j] = cell_type_exprs[i][j] / (local_sum_ptr[i] + 1)
        else:
            for j in prange(cell_type_exprs.shape[1]):
                for i in range(cell_type_exprs.shape[0]):
                    cell_type_exprs[i][j] = cell_type_exprs[i][j] / local_sum_ptr[i]

    free(local_sum_ptr)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _wmean_std_optimized(cnp.float64_t[:] values, double* weights, double* result) nogil:
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

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int maximize_molecule_clusters_norm_optimized(_NormalComponentList* normalComponentsList, cnp.float64_t[:, :] gene_vecs,
                                      cnp.float64_t[:] confidence, cnp.float64_t[:, :] assignment_probs,
                                      add_pseudocount=False) nogil:
    cdef int ci, di, i
    cdef double** c_weights_matrix
    cdef int n_components = normalComponentsList.size
    cdef int n_dists
    cdef double** m_s2_matrix
    cdef double* c_weights_sum

    m_s2_matrix = <double **>malloc(n_components * sizeof(double*))
    for i in range(n_components):
        m_s2_matrix[i] = <double *>malloc(2 * sizeof(double))
    cdef int confidence_shape_zero = confidence.shape[0]
    c_weights_matrix = <double **>malloc(n_components * sizeof(double *))
    for i in range(n_components):
        c_weights_matrix[i] = <double *>malloc(confidence_shape_zero * sizeof(double))
    c_weights_sum = <double *>malloc(n_components * sizeof(double))

    with parallel():
        for ci in prange(n_components):
            for i in range(confidence_shape_zero):
                c_weights_matrix[ci][i] = assignment_probs[ci, i] * confidence[i]

            for di in range(normalComponentsList.data[ci].size):
                _wmean_std_optimized(gene_vecs[:, di], c_weights_matrix[ci], m_s2_matrix[ci])
                updateFNormal(m_s2_matrix[ci][0], m_s2_matrix[ci][1], &normalComponentsList.data[ci].dists[di])

            c_weights_sum[ci] = 0
            for i in range(confidence_shape_zero):
                c_weights_sum[ci] += c_weights_matrix[ci][i]

            normalComponentsList.data[ci].n = c_weights_sum[ci]

    for i in range(n_components):
        free(c_weights_matrix[i])
    free(c_weights_matrix)
    for i in range(n_components):
        free(m_s2_matrix[i])
    free(m_s2_matrix)
    free(c_weights_sum)

    return 0

def prob_array(values, max_value=None, smooth=0.0):
    if max_value is None:
        max_value = max(values) + 1

    sum_value = len(values) + max_value * smooth
    counts = [smooth / sum_value] * max_value
    for v in values:
        counts[v] += 1.0 / sum_value

    return counts

def hash_64_64(n):
    a = n
    a = ~a + (a << 21) & 0xFFFFFFFFFFFFFFFF
    a = a ^ (a >> 24)
    a = (a + (a << 3) + (a << 8)) & 0xFFFFFFFFFFFFFFFF
    a = a ^ (a >> 14)
    a = (a + (a << 2) + (a << 4)) & 0xFFFFFFFFFFFFFFFF
    a = a ^ (a >> 28)
    a = (a + (a << 31)) & 0xFFFFFFFFFFFFFFFF
    return a

def init_cell_type_exprs(
    genes: np.ndarray,
    cell_type_exprs: Optional[Union[np.ndarray, None]] = None,
    assignment: Optional[np.ndarray] = None,
    n_clusters: int = 1,
    init_mod: int = 10000
) -> np.ndarray:
    if cell_type_exprs is not None:
        return copy.deepcopy(cell_type_exprs)

    if n_clusters <= 1:
        if assignment is None:
            raise ValueError("Either n_clusters, assignment, or cell_type_exprs must be specified")
        n_clusters = np.max(assignment)

    if init_mod < 0:
        # Assuming `prob_array` is a function defined elsewhere
        cell_type_exprs = np.concatenate(
            [prob_array(genes == i, max_value=np.max(genes)) for i in range(1, n_clusters + 1)], axis=1
        ).T
    else:
        gene_probs = np.array(prob_array(genes))  # Assuming `prob_array` is a function defined elsewhere
        noise = np.concatenate([[hash_64_64(x1 * (x2 ** 2)) for x2 in range(1, n_clusters+1)] for x1 in range(1, len(gene_probs)+1)]).reshape(gene_probs.shape[0], n_clusters) % init_mod / 100000
        cell_type_exprs = (gene_probs[:, np.newaxis].T * (0.95 + noise.T))
    
    cell_type_exprs = (cell_type_exprs + 1) / (np.sum(cell_type_exprs, axis=1, keepdims=True) + 1)
    return cell_type_exprs

def init_assignment_probs_inner(genes, cell_type_exprs):
    return cell_type_exprs[:, genes]

def init_assignment_probs(assignment, n_clusters):
    assignment_probs = np.zeros((n_clusters, len(assignment)))

    for i, a in enumerate(assignment):
        if a is None:
            assignment_probs[:, i] = 1 / assignment_probs.shape[0]
        else:
            assignment_probs[a - 1, i] = 1.0

    return assignment_probs

def init_assignment_probs_alt(genes, cell_type_exprs, n_clusters=1, assignment=None, assignment_probs=None):
    if assignment_probs is not None:
        return np.copy(assignment_probs)

    if assignment is not None:
        return init_assignment_probs(assignment, n_clusters)

    assignment_probs = init_assignment_probs_inner(genes, cell_type_exprs)

    col_sum = np.sum(assignment_probs, axis=0)
    assignment_probs[:, col_sum < 1e-10] = 1 / assignment_probs.shape[0]

    assignment_probs /= np.sum(assignment_probs, axis=0)

    return assignment_probs

def init_categorical_mixture(genes, cell_type_exprs=None, assignment=None, assignment_probs=None, n_clusters=1, init_mod=10000):
    cell_type_exprs = init_cell_type_exprs(genes, cell_type_exprs, assignment, n_clusters=n_clusters, init_mod=init_mod)
    
    assignment_probs = init_assignment_probs_alt(genes, cell_type_exprs, n_clusters=n_clusters, assignment=assignment, assignment_probs=assignment_probs)

    return cell_type_exprs, assignment_probs

def init_normal_cluster_mixture(gene_vectors, confidence, n_clusters=1, assignment=None, assignment_probs=None):
    if assignment_probs is None:
        if assignment is None:
            assignment = np.empty((gene_vectors.shape[0]), dtype=object)
            assignment.fill(None)
        assignment_probs = init_assignment_probs(assignment, n_clusters)
        assignment_probs += np.random.normal(0.001, 0.001, assignment_probs.shape)
    #componentsPython = [NormalComponent([FNormal(0.0, 1.0) for _ in range(gene_vectors.shape[1])], 1) for _ in range(assignment_probs.shape[0])]

    normalComponentsListSize = assignment_probs.shape[0]
    normalComponentsList = createEmptyNormalComponentList(normalComponentsListSize)
    for i in range(normalComponentsListSize):
        # create normalComponent
        componentSize = gene_vectors.shape[1]
        normalComponent = createEmptyNormalComponent(componentSize)
        normalComponent.n = 1
        for j in range(componentSize):
            # create FNormal
            fNormal = createFNormal(0.0, 1.0)
            normalComponent.dists[j] = fNormal
        normalComponentsList.data[i] = normalComponent
    maximize_molecule_clusters_norm_optimized(normalComponentsList, gene_vectors, confidence, assignment_probs)

    cdef NormalComponentListWrapper wrapper = NormalComponentListWrapper()
    wrapper._c_list = normalComponentsList

    return wrapper, assignment_probs

def init_cluster_mixture(genes, confidence, n_clusters=1, components=None, assignment=None, assignment_probs=None, init_mod=10000, method='categorical'):
    if components is not None and assignment_probs is not None:
        return components, assignment_probs

    if method == 'normal':
        if components is None:
            components, assignment_probs = init_normal_cluster_mixture(genes, confidence, n_clusters, assignment, assignment_probs)
        else:
            if assignment_probs is None and assignment is None:
                raise ValueError("Either assignment or assignment_probs must be provided for method='normal'")
            if assignment_probs is None:
                assignment_probs = init_assignment_probs(assignment, n_clusters)
    elif method == 'categorical':
        components, assignment_probs = init_categorical_mixture(genes, components, assignment, assignment_probs, n_clusters=n_clusters, init_mod=init_mod)
    else:
        raise ValueError(f"Unknown method: {method}")

    return components, assignment_probs

def cluster_molecules_on_mrf_alt(genes, adj_list, confidence, n_clusters=1, tol=0.01, do_maximize=True, max_iters=None, 
                             n_iters_without_update=20, components=None, assignment=None, assignment_probs=None, 
                             verbose=False, progress=None, mrf_weight=1.0, init_mod=10000, method='categorical', **kwargs):
    if max_iters is None:
        max_iters = max(10000, len(genes) // 200)

    adj_weights = [adj_list.weights[i] * confidence[adj_list.ids[i]] for i in range(len(adj_list.ids))]
    adj_list = AdjList(adj_list.ids, adj_weights)
    
    components, assignment_probs = init_cluster_mixture(
        genes, confidence, n_clusters=n_clusters, components=components, assignment=assignment, 
        assignment_probs=assignment_probs, init_mod=init_mod, method=method
    )
    assignment_probs_prev = np.copy(assignment_probs)
    cell_type_exprs = components
    if method == 'categorical':
        max_diffs, change_fracs = cluster_molecules_loop(assignment_probs, assignment_probs_prev, cell_type_exprs, confidence, genes, adj_list, max_iters, tol, n_iters_without_update, do_maximize, mrf_weight)
    elif method == 'normal':
        max_diffs, change_fracs = cluster_molecules_loop_norm(assignment_probs, assignment_probs_prev, cell_type_exprs, confidence, genes, adj_list, max_iters, tol, n_iters_without_update, do_maximize, mrf_weight)
    assignment = np.argmax(assignment_probs, 0)
    return {'exprs': components, 'assignment': assignment, 'diffs': max_diffs, 
            'assignment_probs': assignment_probs, 'change_fracs': change_fracs}