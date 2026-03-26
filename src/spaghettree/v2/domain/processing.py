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
def infer_module_names(
    df: pl.DataFrame,
    module_col: str = "optimised_module",
    inferred_module_col: str = "inferred_module",
) -> pl.DataFrame:
    lf = df.lazy()

    split_module_name = pl.col(module_col).str.split(".")

    best = (
        lf.with_columns(
            split_module_name.list.slice(0, split_module_name.list.len() - 1)
            .list.join(".")
            .alias(inferred_module_col)
        )
        .group_by([module_col, inferred_module_col])
        .agg(pl.len().alias("freq"))
        .sort(by=[module_col, "freq", inferred_module_col], descending=[False, True, False])
        .group_by(module_col)
        .first()
    )

    unique_best = best.sort(by=["freq", inferred_module_col], descending=[True, False]).unique(
        inferred_module_col, keep="first"
    )

    overflow = best.join(unique_best.select(module_col), on=module_col, how="anti").with_columns(
        (pl.col(inferred_module_col) + "_" + pl.col(module_col).str.split(".").list.get(-1)).alias(
            inferred_module_col
        )
    )
    return (
        lf.join(
            pl.concat([unique_best, overflow]).select([module_col, inferred_module_col]),
            on=module_col,
            how="left",
        )
        # .select("node_name", "inferred_module")
        .with_columns(construct_imports())
        .collect()
    )


def construct_imports(
    module_col: str = "inferred_module",
    node_col: str = "node_name",
    import_col: str = "import_stmt",
) -> pl.Expr:
    return (
        "from " + pl.col(module_col) + " import " + pl.col(node_col).str.split(".").list.get(-1)
    ).alias(import_col)
