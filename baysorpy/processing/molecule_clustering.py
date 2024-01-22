import numpy as np

from sklearn.decomposition import FastICA
from scipy.stats import norm
from math import log, pi, exp
from typing import Union, List
from typing import Optional, List, Union
from noise_estimation import AdjList, build_molecule_graph
from molecule_clustering_cython import cluster_molecules_on_mrf_alt

import sys
sys.path.append('../')
from neighborhood_composition import neighborhood_count_matrix, estimate_molecule_vectors

np.random.seed(42)

# # Equivalent to Julia's FNormal struct
# class FNormal:
#     def __init__(self, mu: float, sigma: float):
#         self.mu = mu
#         self.sigma = sigma
#         self.c = -0.5 * log(2 * pi) - log(sigma)
#         self.s = 0.5 / sigma**2

# # Equivalent to Julia's NormalComponent struct
# class NormalComponent:
#     def __init__(self, dists: List[FNormal], n: float):
#         self.dists = dists
#         self.n = n

# # Function to replace normal_logpdf
# def normal_logpdf(n: FNormal, v: float) -> float:
#     return n.c - (v - n.mu)**2 * n.s

# def normal_logpdf_alt(a: float, b: float) -> float:
#     return a * b

# def pdf(comp: NormalComponent, vec: np.ndarray) -> float:
#     dens = 0.0
#     for i in range(len(vec)):
#         dens += normal_logpdf(comp.dists[i], vec[i])
#     return comp.n * exp(dens)

# def comp_pdf(cell_type_exprs, ci, gene_or_factor):
#     if isinstance(cell_type_exprs, np.ndarray):  # Equivalent to CatMixture
#         return cell_type_exprs[ci, gene_or_factor]
#     elif isinstance(cell_type_exprs, list):  # Equivalent to NormMixture
#         return pdf(cell_type_exprs[ci], gene_or_factor)
#     else:
#         raise ValueError("Invalid type for cell_type_exprs")

# def maximize_molecule_clusters(cell_type_exprs, genes, confidence, assignment_probs,
#                                prior_exprs=None, prior_stds=None, add_pseudocount=False):
#     cell_type_exprs.fill(0.0)
    
#     for i in range(len(genes)):
#         t_gene = genes[i]
#         t_conf = confidence[i]
        
#         for j in range(cell_type_exprs.shape[0]):
#             cell_type_exprs[j, t_gene] += t_conf * assignment_probs[j, i]

#     if prior_exprs is not None:
#         mult = np.sum(cell_type_exprs, axis=1, keepdims=True)
#         for i in range(cell_type_exprs.shape[0]):
#             for j in range(cell_type_exprs.shape[1]):
#                 cell_type_exprs[i, j] = adj_value_norm(cell_type_exprs[i, j], 
#                                                        prior_exprs[i, j] * mult[i, 0], 
#                                                        prior_stds[i, j] * mult[i, 0])

#     if add_pseudocount:
#         #cell_type_exprs = (cell_type_exprs + 1) / (np.sum(cell_type_exprs, axis=1, keepdims=True) + 1)
#         cell_type_exprs += 1
#         cell_type_exprs /= (np.sum(cell_type_exprs, axis=1, keepdims=True) + 1)
#     else:
#         cell_type_exprs /= np.sum(cell_type_exprs, axis=1, keepdims=True)

# def process_component_maximize_molecule_clusters_threaded(ci, components, gene_vecs, confidence, assignment_probs):
#     c_weights = assignment_probs[ci, :] * confidence
#     dists = components[ci].dists
#     new_dists = []
#     for di in range(dists):
#         gene_vec = gene_vecs[:, di]
#         # Assuming wmean_std returns parameters for FNormal
#         dist_params = np.zeros(2)
#         dist_params = wmean_std(gene_vec, c_weights, dist_params)
#         new_dists.append(FNormal(*dist_params))
#     components[ci] = NormalComponent(new_dists, np.sum(c_weights))

# def maximize_molecule_clusters_threaded(components, gene_vecs, confidence, assignment_probs, add_pseudocount=False):
#     threads = []
#     for ci in range(len(components)):
#         thread = threading.Thread(target=process_component_maximize_molecule_clusters_threaded, args=(ci, components, gene_vecs, confidence, assignment_probs))
#         threads.append(thread)
#         thread.start()

#     for thread in threads:
#         thread.join()

# Implementation of get_gene_vec function
def get_gene_vec(genes, i):
    if isinstance(genes, np.ndarray):
        if genes.ndim == 1:  # Vector
            return genes[i]
        elif genes.ndim == 2:  # Matrix
            return genes[i, :]
        else:
            raise ValueError("Invalid dimension for genes")
    else:
        raise ValueError("Invalid type for genes")

# def expect_molecule_clusters(
#         assignment_probs: np.ndarray, 
#         assignment_probs_prev: np.ndarray, 
#         cell_type_exprs: Union[np.ndarray, List[NormalComponent]], 
#         genes: Union[np.ndarray, List[int]], 
#         adj_list: AdjList, 
#         mrf_weight: float = 1.0, 
#         only_mrf: bool = False,
#         is_fixed: Optional[np.ndarray] = None
#     ) -> float:
#     total_ll = 0.0
    
#     for i in range(len(adj_list.ids)):
#         #print('assignment_probs')
#         # TODO check with Julia
#         # if is_fixed is not None or not is_fixed[i]:
#         #     continue
    
#         gene = get_gene_vec(genes, i)
#         cur_weights = adj_list.weights[i]  # Assumes weights have been pre-multiplied by confidence
#         cur_points = adj_list.ids[i]

#         dense_sum = 0.0
#         for ri in range(assignment_probs.shape[0]):
#             c_d = 0.0
#             for j in range(len(cur_points)):
#                 a_p = assignment_probs_prev[ri][cur_points[j]]
#                 if a_p > 1e-5:
#                     c_d += cur_weights[j] * a_p
            
#             mrf_prior = np.exp(c_d * mrf_weight)
#             if only_mrf:
#                 assignment_probs[ri, i] = mrf_prior
#             else:
#                 assignment_probs[ri, i] = comp_pdf(cell_type_exprs, ri, gene) * mrf_prior
#             dense_sum += assignment_probs[ri, i]
        
#         if dense_sum > 1e-20:
#             assignment_probs[:, i] /= dense_sum
#             total_ll += np.log10(dense_sum)
#         else:
#             assignment_probs[:, i] = 1 / assignment_probs.shape[0]

#     return total_ll
#@profile        
def cluster_molecules_on_mrf(df_spatial, adj_list, n_clusters, confidence_threshold=0.95, **kwargs):
    # cor_mat = pairwise_gene_spatial_cor(df_spatial['gene'], df_spatial['confidence'], adj_list, confidence_threshold=confidence_threshold)

    ct_exprs_init = None
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('error')
    #     try:
    #         ica = FastICA(n_components=n_clusters, max_iter=10000)
    #         ica_fit = ica.fit_transform(cor_mat)
    #         ct_exprs_init = (np.abs(ica_fit.T) / np.sum(np.abs(ica_fit.T), axis=0)).T
    #     except Exception as e:
    #         print("ICA did not converge, falling back to random initialization.\n")
    #         ct_exprs_init = None
    genes = np.array(df_spatial['gene'])
    confidence = np.array(df_spatial['confidence'])
    if kwargs['method'] == 'normal':
        pos_data = position_data(df_spatial)
        ncm_mat = neighborhood_count_matrix(pos_data, genes, n_clusters)
        genes = estimate_molecule_vectors(ncm_mat, 1)
        print('kek')
    if ct_exprs_init is not None:
        ct_exprs_init = ct_exprs_init.T
    print('end')
    return cluster_molecules_on_mrf_alt(genes, adj_list, confidence, components=ct_exprs_init, n_clusters=n_clusters, **kwargs)

### To do
# def filter_small_molecule_clusters(genes, confidence, adjacent_points, assignment_probs, cell_type_exprs, min_mols_per_cell, confidence_threshold=0.95):
#     # Finding the index of the maximum value in each column of assignment_probs
#     assignment = np.argmax(assignment_probs, axis=0)

#     # Replace this with the actual implementation of get_connected_components_per_label in Python
#     # The output format should match that of the Julia function
#     conn_comps_per_clust = get_connected_components_per_label(
#         assignment, adjacent_points, 1, confidence=confidence, confidence_threshold=confidence_threshold)[0]

#     # Counting molecules in each component per cluster
#     n_mols_per_comp_per_clust = [len(c) for c in conn_comps_per_clust]

#     # Finding cluster IDs meeting the minimum molecules per cell criteria
#     real_clust_ids = [i for i, n_mols in enumerate(map(max, n_mols_per_comp_per_clust)) if n_mols >= min_mols_per_cell]

#     if len(real_clust_ids) == assignment_probs.shape[0]:
#         return assignment_probs, n_mols_per_comp_per_clust, real_clust_ids

#     # Filtering the matrices based on real cluster IDs
#     assignment_probs = assignment_probs[real_clust_ids, :]
#     cell_type_exprs = cell_type_exprs[real_clust_ids, :]

#     # Adjusting assignment probabilities based on cell type expressions
#     for i in np.where(np.sum(assignment_probs, axis=0) < 1e-10)[0]:
#         assignment_probs[:, i] = cell_type_exprs[:, genes[i]]

#     # Normalizing assignment probabilities
#     assignment_probs /= np.sum(assignment_probs, axis=0)

#     return assignment_probs, [n_mols_per_comp_per_clust[i] for i in real_clust_ids], real_clust_ids


## Utils

def position_data(df):
    if 'z' in df.columns:
        return np.array(df[['x', 'y', 'z']])
    return np.array(df[['x', 'y']])

def adj_value_norm(x: float, mu: float, sigma: float) -> float:
    dx = x - mu
    z = abs(dx) / sigma
    if z < 1:
        return x

    if z < 3:
        return mu + np.sign(dx) * (1 + (z - 1) / 4) * sigma

    return mu + np.sign(dx) * (np.sqrt(z) + 1.5 - np.sqrt(3)) * sigma
    
def pairwise_gene_spatial_cor(genes: np.ndarray, confidence: np.ndarray, adj_list: AdjList, confidence_threshold: float = 0.95) -> np.ndarray:
    max_gene = np.max(genes) + 1
    gene_cors = np.zeros((max_gene, max_gene))
    sum_weight_per_gene = np.zeros(max_gene)

    for gi, (g2, conf) in enumerate(zip(genes, confidence)):
        if conf < confidence_threshold:
            continue

        cur_adj_points = adj_list.ids[gi]
        cur_adj_weights = adj_list.weights[gi]

        for ai, adj_point in enumerate(cur_adj_points):
            if confidence[adj_point] < confidence_threshold:
                continue

            g1 = genes[adj_point]
            cw = cur_adj_weights[ai]
            gene_cors[g2 - 1, g1 - 1] += cw  # Adjusting index for 0-based
            sum_weight_per_gene[g1 - 1] += cw
            sum_weight_per_gene[g2 - 1] += cw

    for ci in range(max_gene):
        for ri in range(max_gene):
            gene_cors[ri, ci] /= max(np.sqrt(sum_weight_per_gene[ri] * sum_weight_per_gene[ci]), 0.1)

    return gene_cors

## Wrappers
#@profile
def estimate_molecule_clusters(df_spatial, n_clusters, **kwargs):
    print("Clustering molecules...")
    pos_data = position_data(df_spatial).T

    adj_list = build_molecule_graph(pos_data)
    print('Enter mol clusters')
    mol_clusts = cluster_molecules_on_mrf(df_spatial, adj_list, n_clusters=n_clusters, **kwargs)

    print("Done")
    return mol_clusts