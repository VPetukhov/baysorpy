from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from collections import Counter
from pandas import Series, DataFrame
from scipy import sparse
import numpy as np
from typing import List, Optional

from .baysor_wrappers import _neighborhood_count_matrix_jl


def _neighborhood_count_matrix_py(
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
        return _neighborhood_count_matrix_py(pos_data, gene_ids, k, **kwargs)

    if method == 'jl':
        return _neighborhood_count_matrix_jl(pos_data, gene_ids, k, **kwargs)

    raise ValueError(f"Method {method} not found. Please choose from ['py', 'jl']")


def estimate_gene_vectors(
        neighb_mat, gene_ids: np.ndarray, embedding_size: int, gene_names: Optional[List[str]] = None,
        var_clip: float = 0.95, random_vectors_init=None
    ):
    n_genes = gene_ids.max() + 1
    if random_vectors_init is None: # mostly used for dev purposes
        random_vectors_init = np.random.normal(size=(n_genes, embedding_size))

    coexpr_mat = neighb_mat.T.dot(neighb_mat)
    if not isinstance(coexpr_mat, np.ndarray):
        coexpr_mat = coexpr_mat.A

    if var_clip > 0:
        # Genes that mostly co-variate with themselves don't get updated by other genes
        # Som we clip their covariance, which greatly improves convergence
        diag_vals = np.diag(coexpr_mat)
        total_var = coexpr_mat.sum(axis=0)
        diag_frac = diag_vals / total_var

        coexpr_mat[np.diag_indices_from(coexpr_mat)] = np.minimum(
            (np.quantile(diag_frac, 1 - var_clip) * total_var).astype(int),
            diag_vals
        )

    init_cov = np.random.normal(size=(coexpr_mat.shape[0], embedding_size))
    gene_emb = (coexpr_mat.dot(init_cov).T / coexpr_mat.sum(axis=1)).T

    if gene_names is not None:
        gene_emb = DataFrame(gene_emb, index=gene_names)

    return gene_emb


## Embedding to colors

LC1C2_TO_RGB = np.matrix([
    [ 1.,  0.114,  0.74364896],
    [ 1.,  0.114, -0.41108545],
    [ 1., -0.886,  0.16628176],
])

def _rotate(v: np.array, d: np.array):
    """Rotate the vector."""

    ct = np.cos(d)
    st = np.sin(d)

    return np.c_[
        v[:,0],
        ct * v[:,1] - st * v[:,2],
        st * v[:,1] + ct * v[:,2]
    ]


def _orgb_to_srgb(lcc: np.array):
    # adopted from coloraide https://github.com/facelessuser/coloraide/blob/b9e422bc4eec766848267d2a09b9cbdf54d32ae1/coloraide/spaces/orgb.py
    """Convert oRGB to sRGB."""

    theta0 = np.arctan2(lcc[:,2], lcc[:,1])
    theta = np.array(theta0)
    atheta0 = np.abs(theta0)
    theta[atheta0 < (np.pi / 2)] *= 2/3

    m2 = ((np.pi / 2) <= atheta0) & (atheta0 <= np.pi)
    theta[m2] = np.copysign((np.pi / 3) + (4 / 3) * (atheta0[m2] - np.pi / 2), theta0[m2])

    return np.dot(LC1C2_TO_RGB, _rotate(lcc, theta - theta0).T).T


def _scale_embedding(embedding: np.array, dim_ranges: List[int], clip: float = 0.01):
    emb_scaled = np.array(embedding);
    emb_scaled = np.maximum(emb_scaled, np.quantile(emb_scaled, clip, axis=0))
    emb_scaled = np.minimum(emb_scaled, np.quantile(emb_scaled, 1-clip, axis=0))

    # Rotate the space to maximize variance difference between components
    emb_scaled = PCA().fit_transform(emb_scaled)[:,::-1]
    emb_scaled -= emb_scaled.mean(axis=0)

    # Scale dimensions to preserve their ratio and account for differences in dimension ranges + clipping
    scales = np.abs(emb_scaled).max(axis=0)
    scale = np.max(scales / dim_ranges)
    emb_scaled /= scale

    return emb_scaled


def _embedding_to_orgb(embedding: np.array, l_clip: float = 0.1, val_clip: float = 0.01) -> np.array:
    dim_ranges = [0.5 * (1 - 2 * l_clip), 1, 1] # [0, 1], [-1, -1], [-1, 1]
    emb_scaled = _scale_embedding(embedding, dim_ranges=dim_ranges, clip=val_clip)
    emb_scaled[:,0] -= emb_scaled[:,0].min() - l_clip

    rgb = _orgb_to_srgb(emb_scaled)
    rgb = np.maximum(np.minimum(rgb, 1.0), 0.0)

    return rgb


def _embedding_to_lab(embedding: np.array, l_clip: float = 0.1, val_clip: float = 0.01) -> np.array:
    try:
        import skimage
    except ImportError as e:
        raise Exception("skimage must be installed to use lab color space")

    dim_ranges = [0.5 * 100/128 * (1 - 2 * l_clip), 1, 1] # [0, 100], [-128, 128], [-128, 128]
    emb_scaled = _scale_embedding(embedding, dim_ranges=dim_ranges, clip=val_clip)
    emb_scaled[:,0] -= emb_scaled[:,0].min() - l_clip
    emb_scaled *= 128

    return skimage.color.lab2rgb(emb_scaled)


def embedding_to_color(embedding: np.array, space: str = "orgb", **kwargs):
    """
    Converts an arbitrary 3D `embedding` into a set of RGB colors
    using color space `space` as an intermediate representation.
    """
    if space == "orgb":
        return _embedding_to_orgb(embedding, **kwargs)

    if space == "lab":
        return _embedding_to_lab(embedding, **kwargs)

    raise ValueError("`space` must be one of ['orgb', 'lab']")
