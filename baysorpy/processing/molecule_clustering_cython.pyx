from noise_estimation cimport AdjList

from libc.math cimport log, log10, exp, M_PI, fabs
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

from cython.parallel import parallel, prange
import tqdm

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

cdef class FNormal:
    cdef public double mu, sigma, c, s

    def __init__(self, double mu, double sigma):
        self.mu = mu
        self.sigma = sigma
        self.c = -0.5 * log(2 * M_PI) - log(sigma)
        self.s = 0.5 / sigma**2

cdef class NormalComponent:
    cdef public list dists
    cdef public double n

    def __init__(self, list dists, double n):
        self.dists = dists
        self.n = n

cdef double normal_logpdf(FNormal n, double v) nogil:
    return n.c - (v - n.mu)**2 * n.s

cdef double pdf(NormalComponent comp, double[:] vec):
    cdef double dens = 0.0
    cdef int i
    for i in range(vec.shape[0]):
        dens += normal_logpdf(comp.dists[i], vec[i])
    return comp.n * exp(dens)

#cpdef double comp_pdf(cell_type_exprs, int ci, double[:] gene_or_factor):
#    if isinstance(cell_type_exprs, cnp.ndarray):  # Equivalent to CatMixture
#        return cell_type_exprs[ci, gene_or_factor]
#    elif isinstance(cell_type_exprs, list):  # Equivalent to NormMixture
#        return pdf(cell_type_exprs[ci], gene_or_factor)
#    else:
#        raise ValueError("Invalid type for cell_type_exprs")

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
