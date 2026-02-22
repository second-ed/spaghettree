import polars as pl
from danom import safe

from spaghettree.v2.domain.visitors import NodeMetadata


@safe
def nodes_to_lf(collected_nodes: list[NodeMetadata]) -> pl.DataFrame:
    return entities_to_lf(collected_nodes).pipe(calc_fact_table)


def entities_to_lf(entities: list[NodeMetadata]) -> pl.DataFrame:
    rows = []

    for ent in entities:
        if ent.calls:
            for call in ent.calls:
                if ent.qualified_names:
                    name = ent.qualified_names[0]
                    rows.append(
                        {
                            # "filepath": ent.filepath,
                            "name": name.name,
                            "source": name.source.name,
                            "scope": type(ent.scope).__name__,
                            "call_name": call.name,
                            "call_source": call.source.name,
                        }
                    )
        elif ent.qualified_names:
            name = ent.qualified_names[0]
            rows.append(
                {
                    # "filepath": ent.filepath,
                    "name": name.name,
                    "source": name.source.name,
                    "scope": type(ent.scope).__name__,
                    "call_name": "",
                    "call_source": "",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "name": pl.String(),
            "source": pl.String(),
            "scope": pl.String(),
            "call_name": pl.String(),
            "call_source": pl.String(),
        },
    )


def lf_to_call_tree(df: pl.DataFrame) -> dict[str, list[str]]:
    df = (
        df.group_by("entity_name")
        .agg(pl.when(pl.col("call_name").ne("")).then(pl.col("call_name")))
        .with_columns(pl.col("call_name").list.filter(pl.element().is_not_null()))
    )
    return {row["entity_name"]: row["call_name"] for row in df.to_dicts()}


def calc_fact_table(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("scope").ne("FunctionScope") & pl.col("call_source").ne("BUILTIN"))
        .with_columns(calc_entity_name(), calc_no_locals_call_name())
        .with_columns(
            remove_non_native_calls(),
            calc_module_name(),
            calc_module_name("call_name", "call_module_name"),
        )
        .filter(pl.col("call_name") != pl.col("entity_name"))
    )


def calc_entity_name(name_col: str = "name", scope_col: str = "scope") -> pl.Expr:
    split_col = pl.col(name_col).str.split(".")
    return (
        pl.when(pl.col(scope_col).eq("ClassScope"))
        .then(split_col.list.slice(0, split_col.list.len() - 1).list.join("."))
        .otherwise(pl.col(name_col))
        .alias("entity_name")
    )


def calc_module_name(name_col: str = "entity_name", alias: str = "module_name") -> pl.Expr:
    split_col = pl.col(name_col).str.split(".")
    return split_col.list.slice(0, split_col.list.len() - 1).list.join(".").alias(alias)


def calc_no_locals_call_name(call_name_col: str = "call_name") -> pl.Expr:
    return (
        pl.col(call_name_col)
        .str.split("<locals>")
        .list.get(0)
        .str.strip_chars(".")
        .alias("call_name")
    )


def remove_non_native_calls() -> pl.Expr:
    return (
        pl.when(pl.col("call_name").is_in(pl.col("entity_name").implode()))
        .then(pl.col("call_name"))
        .otherwise(pl.lit(""))
    )
