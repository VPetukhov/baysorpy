from sklearn.metrics import roc_auc_score
from pandas import DataFrame
import numpy as np
import scanpy as sc

def extract_markers_per_type(adata: sc.AnnData, annotation_col: str, append_specificity_metrics: bool = True, top_k: int = 50) -> DataFrame:
    if 'rank_genes_groups' not in adata.uns:
        sc.tl.rank_genes_groups(adata, annotation_col, method='wilcoxon')

    markers_per_type = DataFrame(adata.uns['rank_genes_groups']['names'][:top_k])
    markers_per_type = markers_per_type.stack().reset_index().iloc[:,1:].rename(columns={"level_1": "type", 0: "gene"})

    annot = adata.obs[annotation_col]
    markers_per_type["AUC"] = markers_per_type.apply(
        lambda x: roc_auc_score(annot.values == x['type'], adata[:,x['gene']].X[:,0]), axis=1
    )

    true_neg_mask = np.array([(annot.values != t) for t in markers_per_type['type']]).T
    false_pos_mask = (adata[:,markers_per_type.gene].X > 0.01) & true_neg_mask
    markers_per_type["Specificity"] = true_neg_mask.sum(axis=0) / (true_neg_mask.sum(axis=0) + false_pos_mask.sum(axis=0))

    return markers_per_type
