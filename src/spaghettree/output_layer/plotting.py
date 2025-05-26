from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from returns.result import safe

if TYPE_CHECKING:
    from spaghettree.__main__ import OptResult


@safe
def get_color_map(module_names: tuple) -> dict[str, np.ndarray]:
    n = int(np.ceil(len(module_names) / 3))
    reds = plt.cm.Reds(np.linspace(0.2, 0.8, n))
    blues = plt.cm.Blues(np.linspace(0.2, 0.8, n))
    greens = plt.cm.Greens(np.linspace(0.2, 0.8, n))
    colors = np.vstack([reds, blues, greens])[::-1]
    return dict(zip(module_names, colors))


def plot_graph(
    path: str,
    adj_mat: np.ndarray,
    nodes: list,
    color_map: dict[str, np.ndarray],
    delim: str = ".",
    figsize: tuple[int, int] = (24, 16),
) -> None:
    adj_df = pd.DataFrame(adj_mat, columns=nodes, index=nodes)
    plt.figure(figsize=figsize)

    G = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph)
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
        arrowsize=2,
        font_size=8,
    )
    nx.draw_networkx_edges(G, pos, node_size=4000)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@safe
def plot_heatmap(
    path: str,
    adj_mat: np.ndarray,
    nodes: list,
    annot: bool = False,
    figsize: tuple[int, int] = (24, 16),
) -> Literal[True]:
    adj_df = pd.DataFrame(adj_mat, columns=nodes, index=nodes)
    plt.figure(figsize=figsize)
    nonzero_rows = adj_df.any(axis=1)
    nonzero_cols = adj_df.any(axis=0)
    adj_df_filtered = adj_df.loc[nonzero_rows, nonzero_cols]
    sns.heatmap(
        adj_df_filtered,
        cmap=sns.color_palette("vlag", as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def bootstrap_column(series: pd.Series, sims: int = 1000) -> np.array[np.floating]:
    return np.array(
        [np.mean(np.random.choice(series, len(series), replace=True)) for _ in range(sims)]
    )


@safe
def save_demeaned_control_test(
    path: str, df: pd.DataFrame, col: str, sims: int = 1000, figsize: tuple[int, int] = (7, 5)
) -> None:
    observed = bootstrap_column(df[col], sims=sims)
    null = bootstrap_column(df[col] - df[col].mean(), sims=sims)

    min_obs_value = np.min(observed)
    p_val = np.mean(null > min_obs_value)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=figsize)

    sns.histplot(observed, label="observed")
    sns.histplot(null, label="demeaned")
    plt.title(f"Demeaned control test: {col} | p-value={p_val:.3f}")
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


@safe
def save_replicates(path: str, package_result: OptResult):
    plt.figure(figsize=(8, 6))

    all_randomised_values = []
    all_randomised_values.extend(package_result.permutates)
    all_randomised_values.extend(package_result.replicates)
    all_randomised_values.extend(package_result.unique_replicates)

    plt.axvline(package_result.best_score, label=package_result.method, color="k", linewidth=2)
    sns.histplot(all_randomised_values, bins="sqrt", label="randomised_values")
    plt.legend()
    plt.title(
        f"Observed vs randomised replicates for {package_result.package} {package_result.method}"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
