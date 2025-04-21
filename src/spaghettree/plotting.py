from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from returns.result import Result, safe


@safe
def get_color_map(module_names: tuple) -> Result[dict, Exception]:
    n = int(np.ceil(len(module_names) / 3))
    reds = plt.cm.Reds(np.linspace(0.2, 1, n))
    blues = plt.cm.Blues(np.linspace(0.2, 1, n))
    greens = plt.cm.Greens(np.linspace(0.2, 1, n))
    colors = np.vstack([reds, blues, greens])
    return dict(zip(module_names, colors))


def plot_graph(
    adj_mat: np.array, nodes: list, color_map: dict[str, np.array], delim: str = "."
) -> None:
    adj_df = pd.DataFrame(adj_mat, columns=nodes, index=nodes)
    plt.figure(figsize=(24, 16))

    G = nx.from_pandas_adjacency(adj_df)
    G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.forceatlas2_layout(
        G,
        scaling_ratio=2,
        strong_gravity=True,
        dissuade_hubs=True,
    )

    node_colors = [color_map[n.split(delim)[0]] for n in G.nodes]
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_size=4000,
        node_shape="o",
        node_color=node_colors,
        edge_color="gray",
        arrows=True,
        font_size=8,
    )


@safe
def plot_heatmap(adj_mat: np.array, nodes: list):
    adj_df = pd.DataFrame(adj_mat, columns=nodes, index=nodes)
    plt.figure(figsize=(24, 16))
    nonzero_rows = adj_df.any(axis=1)
    nonzero_cols = adj_df.any(axis=0)
    adj_df_filtered = adj_df.loc[nonzero_rows, nonzero_cols]
    sns.heatmap(
        adj_df_filtered,
        cmap=sns.color_palette("vlag", as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        # annot=True
    )
    return True
