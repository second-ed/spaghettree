import polars as pl
from danom import safe

from spaghettree.v2.domain.optimisation import AdjMat
from spaghettree.v2.domain.utils import to_df, to_lf


@safe
def compose_facts_and_adj_mat(lf: pl.LazyFrame, adj_mat: AdjMat) -> pl.DataFrame:
    nodes = adj_mat.nodes.pipe(to_lf)

    return (
        adj_mat.communities.pipe(to_lf)
        .join(nodes, left_on="node", right_on="idx")
        .rename({"node_right": "node_name"})
        .join(nodes, left_on="module", right_on="idx")
        .rename({"node_right": "optimised_module"})
        .join(lf, left_on="node_name", right_on="name")
        .pipe(to_df)
    )


@safe
def infer_module_names(
    df: pl.DataFrame, module_col: str = "optimised_module", inferred_module_col: str = "inferred_module"
) -> pl.DataFrame:
    lf = df.pipe(to_lf)

    split_module_name = pl.col(module_col).str.split(".")

    best = (
        lf.with_columns(
            split_module_name.list.slice(0, split_module_name.list.len() - 1).list.join(".").alias(inferred_module_col)
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
        (pl.col(inferred_module_col) + "_" + pl.col(module_col).str.split(".").list.get(-1)).alias(inferred_module_col)
    )
    return (
        lf.join(pl.concat([unique_best, overflow]).select([module_col, inferred_module_col]), on=module_col, how="left")
        # .select("node_name", "inferred_module")
        .with_columns(construct_imports())
        .pipe(to_df)
    )


def construct_imports(
    module_col: str = "inferred_module", node_col: str = "node_name", import_col: str = "import_stmt"
) -> pl.Expr:
    return ("from " + pl.col(module_col) + " import " + pl.col(node_col).str.split(".").list.get(-1)).alias(import_col)
