from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

def plot_molecules(
        df: DataFrame, color: Union[List[str], str] = None, genes: List[str] = None, gene_names: List[str] = None,
        annotation: Union[List[str], str] = None, noise_annotation: str = None, noise_color: str = 'black',
        s=1, alpha=0.1, figsize=(10, 10), ax=None, show_ticks: bool = False,
        legend_loc: str = 'best', legend_markerscale: float = 2.0, **kwargs
    ):

    if isinstance(color, str):
        color = df[color]

    if isinstance(annotation, str):
        annotation = df[annotation]

    if genes is not None:
        if gene_names is None:
            raise ValueError("If genes argument is provided, gene_names has to be provided, as well")

        df = df[np.in1d(gene_names[df.gene], genes)]
        annotation = gene_names[df.gene]

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    if annotation is None:
        if color is None:
            ax.scatter(df.x, df.y, s=s, alpha=alpha, **kwargs)
        else:
            ax.scatter(df.x, df.y, c=color, s=s, alpha=alpha, **kwargs)
    else:
        ann_levels = np.unique(annotation)
        ann_levels = [noise_annotation] + sorted(np.setdiff1d(ann_levels, noise_annotation))
        for l in ann_levels:
            mask = (annotation == l)
            cs = s[mask] if isinstance(s, np.ndarray) else s
            if (noise_annotation is not None) and (l == noise_annotation):
                plt.scatter(df.x.values[mask], df.y.values[mask], s=cs / 4, alpha=alpha / 2, label=l, color=noise_color)
            else:
                plt.scatter(df.x.values[mask], df.y.values[mask], s=cs, alpha=alpha, label=l)
        plt.legend(loc=legend_loc, markerscale=legend_markerscale)

    ax.set_xlim(df.x.min(), df.x.max()); ax.set_ylim(df.y.min(), df.y.max());
    if not show_ticks:
        ax.set_xticks([]); ax.set_yticks([])

    return ax
