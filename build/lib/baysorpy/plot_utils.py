from pandas import DataFrame
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from typing import List, Union, Optional

def plot_molecules(
        df: DataFrame, polygons: Optional[List[List[float]]] = None, color: Optional[Union[List[str], str]] = None,
        genes: Optional[List[str]] = None, gene_names: Optional[List[str]] = None, scale_z: bool = False,
        annotation: Optional[Union[List[str], str]] = None, noise_annotation: Optional[str] = None,
        noise_color: str = 'black', annotation_cmap: str = "tab20",
        s=1, alpha=0.1, figsize=(10, 10), ax=None, show_ticks: bool = False,
        legend_loc: str = 'best', legend_markerscale: float = 2.0, legend_title: Optional[str] = None, **kwargs
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

    if scale_z:
        s = df.z.values * s

    if annotation is None:
        if color is None:
            ax.scatter(df.x, df.y, s=s, alpha=alpha, **kwargs)
        else:
            ax.scatter(df.x, df.y, c=color, s=s, alpha=alpha, **kwargs)
    else:
        ann_levels = np.unique(annotation)
        ann_levels = [noise_annotation] + sorted(np.setdiff1d(ann_levels, noise_annotation))

        cmap = matplotlib.colormaps[annotation_cmap]
        cnorm = plt.Normalize(0, len(ann_levels))

        for i,l in enumerate(ann_levels):
            mask = (annotation == l)
            cs = s[mask] if isinstance(s, np.ndarray) else s
            if (noise_annotation is not None) and (l == noise_annotation):
                plt.scatter(df.x.values[mask], df.y.values[mask], s=cs / 4, alpha=alpha / 2, label=l, color=noise_color)
            else:
                plt.scatter(df.x.values[mask], df.y.values[mask], s=cs, alpha=alpha, label=l, color=cmap(cnorm(i)))
        plt.legend(loc=legend_loc, markerscale=legend_markerscale, title=legend_title)

    if polygons is not None:
        ax.add_collection(LineCollection(polygons, color="black"))

    ax.set_xlim(df.x.min(), df.x.max()); ax.set_ylim(df.y.min(), df.y.max());
    if not show_ticks:
        ax.set_xticks([]); ax.set_yticks([])

    return ax


def rgb_to_hex(rgb_colors: np.ndarray) -> List[str]:
    if isinstance(rgb_colors, np.matrix):
        rgb_colors = rgb_colors.A
    return ["#{:02X}{:02X}{:02X}".format(r[0], r[1], r[2]) for r in np.round(rgb_colors * 255).astype(int)]
    # return [matplotlib.colors.to_hex(c) for c in rgb_colors] # This is 5 times slower
