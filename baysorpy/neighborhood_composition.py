from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.base import BaseEstimator
from umap import ParametricUMAP
from collections import Counter
from pandas import Series, DataFrame
from scipy import sparse
import numpy as np
from typing import List, Optional

from .baysor_wrappers import _neighborhood_count_matrix_jl


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def rgb_vec_to_hex(rgb_vec: np.ndarray) -> np.ndarray:
    hex_colors = (rgb_vec * 255).astype(np.uint8)
    hex_colors = [rgb_to_hex(color) for color in hex_colors]
    return np.array(hex_colors)


def _neighborhood_count_matrix_py(
        pos_data: np.ndarray, gene_ids: np.ndarray, k: int,
        include_self: bool = False, n_genes: Optional[int] = None, max_dist: float = -1.0, n_jobs: int = -1
    ):
    if n_genes is None:
        n_genes = np.max(gene_ids) + 1

    adj_mat = kneighbors_graph(pos_data, k, mode='distance', include_self=include_self, n_jobs=n_jobs)
    if max_dist > 0:
        adj_mat[adj_mat > max_dist] = 0

    nzi,nzj = adj_mat.nonzero();

    gene_counts = sparse.csr_matrix(
        (np.ones(len(nzi), dtype=int), (nzi, gene_ids[nzj])), 
        shape=(pos_data.shape[0], n_genes)
    )
    return gene_counts


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
        neighb_mat, embedding_size: int, gene_names: Optional[List[str]] = None,
        var_clip: float = 0.05, random_vectors_init=None
    ):
    """
    Estimate low-dimensioanl gene vectors from the neighborhood count matrix using Random Indexing algorithm.

    Args:
        neighb_mat: Neighborhood count matrix of shape (n_cells, n_genes)
        embedding_size: Number of dimensions to embed genes into (30-50 is recommended)
        gene_names: Gene names. If provided, the output is a DataFrame with gene names as index. (1D numpy array of shape (n_molecules,))
        var_clip: Fraction of variance to clip from the diagonal of the covariance matrix (improves convergence)
        random_vectors_init: Random vectors to initialize the embedding with (mostly used for dev purposes)
    """
    if random_vectors_init is None: # mostly used for dev purposes
        random_vectors_init = np.random.normal(size=(neighb_mat.shape[1], embedding_size))

    coexpr_mat = neighb_mat.T.dot(neighb_mat)
    if not isinstance(coexpr_mat, np.ndarray):
        coexpr_mat = coexpr_mat.toarray()

    if var_clip > 0:
        # Genes that mostly co-variate with themselves don't get updated by other genes
        # So, we clip their covariance, which greatly improves convergence
        diag_vals = np.diag(coexpr_mat)
        total_var = coexpr_mat.sum(axis=0)
        diag_frac = diag_vals / total_var

        coexpr_mat[np.diag_indices_from(coexpr_mat)] = np.minimum(
            (np.quantile(diag_frac, 1 - var_clip) * total_var).astype(int),
            diag_vals
        )

    gene_emb = (coexpr_mat.dot(random_vectors_init).T / coexpr_mat.sum(axis=1)).T

    if gene_names is not None:
        gene_emb = DataFrame(gene_emb, index=gene_names)

    return gene_emb


def estimate_molecule_vectors(
        neighb_mat, embedding_size: int, var_clip: float = 0.05, pca: bool = True, return_gene_vectors: bool = False
    ):
    """
    Estimate low-dimensioanl molecule vectors from the neighborhood count matrix using Random Indexing algorithm.

    Args:
        neighb_mat: Neighborhood count matrix of shape (n_cells, n_genes)
        embedding_size: Number of dimensions to embed genes into (30-50 is recommended)
        var_clip: Fraction of variance to clip from the diagonal of the covariance matrix (improves convergence)
        pca: Whether to apply PCA to the output vectors. PCA leads to more interpretable dimensions, ordered by their variance.
        return_gene_vectors: Whether to return gene vectors as well
    """
    ri_size = embedding_size if not pca else (2 * embedding_size)
    gene_emb = estimate_gene_vectors(neighb_mat, embedding_size=ri_size, var_clip=var_clip)
    mol_vectors = neighb_mat.dot(gene_emb)

    if pca:
        mol_vectors = PCA(n_components=embedding_size).fit_transform(mol_vectors)

    if return_gene_vectors:
        return mol_vectors, gene_emb

    return mol_vectors


def estimate_molecule_embedding_full(mol_vectors: np.ndarray, estimator: BaseEstimator = None, train_size: int = 50000):
    estimator = estimator or ParametricUMAP(n_components=3)
    # select ids uniformly across the first two PCs:
    if (train_size > 0) & (train_size < mol_vectors.shape[0]):
        train_ids = np.argsort(mol_vectors[:,:2].sum(axis=1))[np.linspace(0, mol_vectors.shape[0] - 1, train_size, dtype=int)]
        train_set = mol_vectors[train_ids,:]
    else:
        train_set = mol_vectors

    estimator.fit(train_set)
    mol_emb = estimator.transform(mol_vectors)
    return mol_emb


def estimate_molecule_embedding_fast(
        gene_vectors: np.ndarray, neighb_mat: np.ndarray, 
        estimator: BaseEstimator = None, n_jobs: int = 1
    ):
    """
    Estimate molecule embedding using pre-computed gene vectors.

    Args:
        gene_vectors: Gene vectors of shape (n_genes, embedding_size)
        neighb_mat: Neighborhood count matrix of shape (n_cells, n_genes)
        estimator: Estimator to use for embedding of gene vectors.
                   By default, MDS is used.
        n_jobs: Number of jobs to run in parallel. Default: 1. 
                *On my test increasing it only slows down the computation.*

    Returns:
        Molecule embedding of shape (n_cells, 3)
    """
    estimator = estimator or MDS(n_components=3, n_jobs=n_jobs, normalized_stress="auto")
    gene_emb = estimator.fit_transform(gene_vectors)
    mol_emb = neighb_mat.dot(gene_emb)
    return mol_emb


def estimate_molecule_colors(
        neighb_mat, embedding_size: int = 30, use_gene_vectors: bool = True,
        estimator: BaseEstimator = None, train_size: int = 50000, verbose: bool = False, **kwargs
    ):
    """
    Estimate molecule colors from the neighborhood count matrix.

    Args:
        neighb_mat: Neighborhood count matrix of shape (n_cells, n_genes)
        embedding_size: Number of dimensions to embed genes into (30-50 is recommended)
        use_gene_vectors: If True, the method embeds gene vectors first and then project molecules onto them.
                          If False, the method embeds molecule vectors directly. Embedding gene vectors is much faster,
                          but in some (rare) cases may lead to worse results.
        estimator: Estimator to use for embedding of gene or molecule vectors.
                   By default, ParametricUMAP is used for molecule vectors and MDS for gene vectors.
                   *For molecules, the estimator must have `fit` and `transform` methods. For genes, only `fit_transform` method.
        train_size: Number of molecules to use for training the estimator for molecule vector embedding. If 0, all molecules are used.
        **kwargs: Additional arguments to pass to the color space conversion function
    """

    if verbose:
        print("Estimating molecule vectors...")

    mol_vectors, gene_vectors = estimate_molecule_vectors(neighb_mat, embedding_size=embedding_size, return_gene_vectors=True)

    if use_gene_vectors:
        if verbose:
            print("Estimating molecule embedding using gene vectors...")

        mol_emb = estimate_molecule_embedding_fast(gene_vectors, neighb_mat, estimator=estimator)
    else:
        if verbose:
            print("Estimating molecule embedding using molecule vectors...")

        mol_emb = estimate_molecule_embedding_full(mol_vectors, estimator=estimator, train_size=train_size)

    if verbose:
        print("Converting embedding to colors...")

    return embedding_to_color(mol_emb, **kwargs)


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

    Args:
        embedding: 3D embedding of shape (n_molecules, 3)
        space: Color space to use as an intermediate representation. One of ['orgb', 'lab']
               'orgb' is less perceptionally uniform, but gives more accurate representation of the original embedding
        **kwargs: Additional arguments to pass to the color space conversion function
    """
    if space == "orgb":
        return _embedding_to_orgb(embedding, **kwargs)

    if space == "lab":
        return _embedding_to_lab(embedding, **kwargs)

    raise ValueError("`space` must be one of ['orgb', 'lab']")
