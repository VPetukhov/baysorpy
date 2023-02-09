from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_molecules(
        df: DataFrame, color: list = None, genes: List[str] = None, gene_names: List[str] = None, annotation: list = None,
        s=1, alpha=0.1, figsize=(10, 10), ax=None, **kwargs
    ):

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
        for l in sorted(np.unique(annotation)):
            mask = (annotation == l)
            plt.scatter(df.x.values[mask], df.y.values[mask], s=s, alpha=alpha, label=l)
        plt.legend()

    ax.set_xlim(df.x.min(), df.x.max()); ax.set_ylim(df.y.min(), df.y.max());
    return ax
