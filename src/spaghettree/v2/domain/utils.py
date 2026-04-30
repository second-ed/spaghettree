import polars as pl


def to_df(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def to_lf(lf: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return lf.lazy() if isinstance(lf, pl.DataFrame) else lf
