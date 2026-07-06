from pathlib import Path

import libcst.matchers as m
import polars as pl
from danom import safe

from spaghettree.v2.domain.utils import to_df, to_lf
from spaghettree.v2.domain.visitors import NodeCollector, NodeMetadata, get_manager

ENTITY_MATCHERS = m.OneOf(m.FunctionDef(), m.ClassDef(), m.Assign(), m.AnnAssign())
REFERENCE_MATCHERS = m.OneOf(m.Call(), m.Name())
IMPORT_MATCHERS = m.OneOf(m.Import(), m.ImportFrom())


@safe
def collect_node_metadata(
    root: str,
    entity_matchers: m.OneOf = ENTITY_MATCHERS,
    reference_matchers: m.OneOf = REFERENCE_MATCHERS,
    import_matchers: m.OneOf = IMPORT_MATCHERS,
) -> list[NodeMetadata]:
    paths = sorted(map(str, Path(root).rglob("**/*.py")))

    manager = get_manager(root, paths)
    manager.resolve_cache()

    collected_nodes = []

    for path in paths:
        wrapper = manager.get_metadata_wrapper_for_path(path)
        collector = NodeCollector(
            entity_matchers=entity_matchers,
            reference_matchers=reference_matchers,
            import_matchers=import_matchers,
            filepath=path,
        )
        wrapper.visit(collector)

        for ent in collector.entities:
            ent.imports.extend(collector.imports)

        collected_nodes.extend(collector.entities)
    return collected_nodes


@safe
def nodes_to_lf(collected_nodes: list[NodeMetadata]) -> pl.LazyFrame:
    return entities_to_lf(collected_nodes).pipe(calc_fact_table)


def entities_to_lf(entities: list[NodeMetadata]) -> pl.LazyFrame:
    rows = []

    for ent in entities:
        imports = [imp.to_dict() for imp in ent.imports]
        if ent.qualified_names:
            name = ent.qualified_names[0]

            rows.append(
                {
                    # "filepath": ent.filepath,  # noqa: ERA001
                    "name": name.name,
                    "source": name.source.name,
                    "scope": type(ent.scope).__name__,
                    "calls": [{"call_name": call.name, "call_source": call.source.name} for call in ent.calls],
                    "imports": imports,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "name": pl.String(),
            "source": pl.String(),
            "scope": pl.String(),
            "calls": pl.List(pl.Struct({"call_name": pl.String(), "call_source": pl.String()})),
            "imports": pl.List(
                pl.Struct(
                    {
                        "import_module": pl.String(),
                        "import_type": pl.String(),
                        "import_name": pl.String(),
                        "import_as_name": pl.String(),
                    }
                )
            ),
        },
    ).pipe(to_lf)


def lf_to_call_tree(
    df: pl.LazyFrame, entity_name: str = "entity_name", call_name: str = "call_name"
) -> dict[str, list[str]]:
    df = (
        df.group_by(entity_name)
        .agg(pl.when(pl.col(call_name).ne("")).then(pl.col(call_name)))
        .with_columns(pl.col(call_name).list.filter(pl.element().is_not_null()))
    )
    return {row[entity_name]: row[call_name] for row in df.pipe(to_df).to_dicts()}


def calc_fact_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.explode("calls")
        .unnest("calls")
        .fill_null("")
        .filter(pl.col("scope").ne("FunctionScope") & pl.col("call_source").ne("BUILTIN"))
        .with_columns(calc_entity_name(), calc_no_locals_call_name())
        .with_columns(remove_non_native_calls(), calc_module_name(), calc_module_name("call_name", "call_module_name"))
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
    return pl.col(call_name_col).str.split("<locals>").list.get(0).str.strip_chars(".").alias("call_name")


def remove_non_native_calls() -> pl.Expr:
    return (
        pl.when(pl.col("call_name").is_in(pl.col("entity_name").implode()))
        .then(pl.col("call_name"))
        .otherwise(pl.lit(""))
    )
