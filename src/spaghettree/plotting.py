from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import Set1
from returns.result import Result, safe


@safe
def get_color_map(calls: pd.DataFrame) -> Result[dict, Exception]:
    colors = Set1(range(calls["module"].nunique()))
    return dict(zip(calls["module"].unique().tolist(), colors))


def plot_graph(adj_df: pd.DataFrame, color_map: dict[str, np.array]) -> None:
    plt.figure(figsize=(12, 8))

    G = nx.from_pandas_adjacency(adj_df)
    G.remove_nodes_from(list(nx.isolates(G)))

    pos = nx.forceatlas2_layout(
        G,
        scaling_ratio=2,
        strong_gravity=True,
        dissuade_hubs=True,
    )

    node_colors = [color_map[n.split(".")[0]] for n in G.nodes]
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
def plot_heatmap(adj_df: pd.DataFrame):
    nonzero_rows = adj_df.any(axis=1)
    nonzero_cols = adj_df.any(axis=0)
    adj_df_filtered = adj_df.loc[nonzero_rows, nonzero_cols]
    sns.heatmap(adj_df_filtered, annot=True)
    return True
