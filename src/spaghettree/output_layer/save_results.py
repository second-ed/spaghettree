from __future__ import annotations

import datetime as dt
import os

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from returns.result import safe

import spaghettree.optimisation_layer.processing as proc
import spaghettree.output_layer.plotting as plot


def write_yaml(data: dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def save_results(results: dict):
    now = dt.datetime.now().strftime(format="%y%m%d_%H%M")

    for name, result in results.items():
        package = result.package
        os.makedirs(f"./results/{now}/{package}/plots", exist_ok=True)

        res_obj = attrs.asdict(result)

        for key, attrib in res_obj.items():
            if isinstance(attrib, pd.DataFrame):
                if key == "epochs":
                    save_plot(
                        f"./results/{now}/{package}/plots/{name}_epochs.png",
                        attrib,
                        name,
                    )
                if key == "original_fact_df":
                    save_graph_plots(f"./results/{now}/{package}/plots/{package}_unopt", attrib)
                if key == "search_df":
                    save_graph_plots(f"./results/{now}/{package}/plots/{name}_opt", attrib)

                res_obj[key] = attrib.to_dict("records")

        write_yaml(res_obj, f"./results/{now}/{package}/{name}.yaml")


@safe
def save_plot(path: str, df: pd.DataFrame, name: str):
    plt.figure()
    for col in df.columns:
        sns.lineplot(
            x=df.index,
            y=df[col],
            label=col,
            linewidth=2 if col == "best_score" else 1,
            color="k" if col == "best_score" else "b",
            alpha=1.0 if col == "best_score" else 0.5,
        )

    plt.title(name.title().replace("_", " "))
    plt.legend()
    plt.ylabel("Fitness Score")
    plt.xlabel("Epoch")
    plt.savefig(path, dpi=300, bbox_inches="tight")


@safe
def save_graph_plots(
    path: str,
    call_df: pd.DataFrame,
    delim: str = ".",
):
    modules, classes, funcs, calls = proc.get_np_arrays(call_df)
    full_func_addr, full_call_addr = proc.clean_calls(modules=modules, classes=classes, funcs=funcs, calls=calls)
    adj_mat, nodes = proc.get_adj_matrix(full_func_addr, full_call_addr)
    communities = np.array([col.split(delim)[0] for col in nodes])
    community_mat = communities[:, None] == communities[None, :]

    x = int(len(np.unique(modules)) * 3.5)
    y = np.ceil(x * 0.7)

    color_map = plot.get_color_map(np.unique(modules)).unwrap()
    plot.plot_graph(
        f"{path}_graph.png",
        adj_mat,
        np.array([node.replace(".", "\n") for node in nodes]),
        color_map=color_map,
        delim="\n",
        figsize=(x, y),
    )

    internal_calls = np.where(community_mat, 1, -1)
    adj_mat = adj_mat * internal_calls

    plot.plot_heatmap(
        f"{path}_heatmap.png",
        adj_mat,
        nodes,
        annot=True,
        figsize=(x, y),
    )
