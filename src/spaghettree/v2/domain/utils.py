from __future__ import annotations

import libcst as cst
import polars as pl


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node: cst.CSTNode) -> str:
    return cst.Module([]).code_for_node(node)


def to_df(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def to_lf(lf: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return lf.lazy() if isinstance(lf, pl.DataFrame) else lf
