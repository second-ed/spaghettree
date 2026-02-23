import polars as pl
from danom import safe

from spaghettree.v2.domain.optimisation import AdjMat


@safe
def compose_facts_and_adj_mat(df: pl.DataFrame, adj_mat: AdjMat) -> pl.DataFrame:
    return (
        adj_mat.communities.join(adj_mat.nodes, left_on="node", right_on="idx")
        .rename({"node_right": "node_name"})
        .join(adj_mat.nodes, left_on="module", right_on="idx")
        .rename({"node_right": "optimised_module"})
        .join(df, left_on="node_name", right_on="name")
    )


@safe
def infer_module_names(df: pl.DataFrame) -> pl.DataFrame:
    lf = df.lazy()

    split_module_name = pl.col("optimised_module").str.split(".")

    best = (
        lf.with_columns(
            split_module_name.list.slice(0, split_module_name.list.len() - 1)
            .list.join(".")
            .alias("parent")
        )
        .group_by(["optimised_module", "parent"])
        .agg(pl.len().alias("freq"))
        .sort(
            by=["optimised_module", "freq", "parent"],
            descending=[False, True, False],
        )
        .group_by("optimised_module")
        .first()
    )

    unique_best = best.sort(by=["freq", "parent"], descending=[True, False]).unique(
        subset=["parent"], keep="first"
    )

    overflow = best.join(
        unique_best.select("optimised_module"),
        on="optimised_module",
        how="anti",
    ).with_columns(
        (pl.col("parent") + "_" + pl.col("optimised_module").str.split(".").list.get(-1)).alias(
            "parent"
        )
    )
    return (
        lf.join(
            pl.concat([unique_best, overflow]).select(["optimised_module", "parent"]),
            on="optimised_module",
            how="left",
        )
        .rename({"parent": "inferred_module"})
        # .select("node_name", "inferred_module")
        .collect()
    )
