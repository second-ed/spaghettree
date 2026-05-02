from __future__ import annotations

from typing import Self

import attrs
import polars as pl
from danom import safe

from spaghettree.v2.domain.utils import to_df


@attrs.define
class AdjMat:
    nodes: pl.DataFrame
    modules: pl.DataFrame
    communities: pl.DataFrame
    dwm: DirectedWeightedModularity

    @classmethod
    @safe
    def from_lf(cls, lf: pl.LazyFrame, *, optimise: bool = True) -> Self:
        nodes = (
            pl.concat(
                [
                    lf.select(pl.col("entity_name").alias("node")),
                    lf.filter(pl.col("call_name").ne("")).select(pl.col("call_name").alias("node")),
                ]
            )
            .unique()
            .sort(by="node")
            .with_row_index("idx")
        )

        modules = (
            lf.select(pl.col("module_name").alias("module"))
            .unique()
            .sort(by="module")
            .with_row_index("idx")
        )

        edges = (
            lf.join(nodes, left_on="entity_name", right_on="node")
            .rename({"idx": "src"})
            .join(nodes, left_on="call_name", right_on="node")
            .rename({"idx": "dst"})
            .group_by(["src", "dst"])
            .len()
            .rename({"len": "weight"})
        )

        communities = (
            lf.join(nodes, left_on="entity_name", right_on="node")
            .rename({"idx": "node"})
            .join(modules, left_on="module_name", right_on="module")
            .rename({"idx": "module"})
            .select("node", "module")
            .unique(maintain_order=True)
            .pipe(to_df)
        )

        dwm = DirectedWeightedModularity.from_edges(edges.pipe(to_df))

        print(f"Pre-optimisation DWM: {dwm.calc(communities)}")  # noqa: T201

        if optimise:
            communities = communities.with_columns(pl.col("node").alias("module"))

        return cls(
            nodes=nodes.pipe(to_df), modules=modules.pipe(to_df), communities=communities, dwm=dwm
        )


@attrs.define(frozen=True)
class DirectedWeightedModularity:
    weighted_nodes: pl.DataFrame
    weighted_edges: pl.DataFrame
    total_edges: float

    @classmethod
    def from_edges(cls, edges: pl.DataFrame) -> Self:
        total_edges = edges.select(pl.col("weight").sum()).item()

        out_deg = edges.group_by("src").agg(pl.col("weight").sum().alias("k_out")).lazy()
        in_deg = edges.group_by("dst").agg(pl.col("weight").sum().alias("k_in")).lazy()

        weighted_nodes = (
            edges.select(pl.col("src").alias("node"))
            .vstack(edges.select(pl.col("dst").alias("node")))
            .lazy()
            .unique()
            .join(out_deg.rename({"src": "node"}), on="node", how="left")
            .join(in_deg.rename({"dst": "node"}), on="node", how="left")
            .with_columns([pl.col("k_out").fill_null(0.0), pl.col("k_in").fill_null(0.0)])
        )

        weighted_edges = (
            edges.lazy()
            .filter(pl.col("weight") != 0)
            .select(pl.col("src"), pl.col("dst"), pl.col("weight").cast(pl.Float64))
        )

        return cls(
            weighted_nodes=weighted_nodes.collect(),
            weighted_edges=weighted_edges.collect(),
            total_edges=float(total_edges),
        )

    def calc(self, communities: pl.DataFrame) -> float:
        if self.total_edges == 0:
            return 0.0

        nodes = self.weighted_nodes.join(communities, on="node", how="left")

        return (
            nodes.join(nodes, how="cross", suffix="_j")
            .join(
                self.weighted_edges, left_on=["node", "node_j"], right_on=["src", "dst"], how="left"
            )
            .with_columns(pl.col("weight").fill_null(0.0))
            .filter(pl.col("module") == pl.col("module_j"))
            .with_columns(
                (pl.col("weight") - (pl.col("k_out") * pl.col("k_in_j") / self.total_edges)).alias(
                    "contrib"
                )
            )
            .select((pl.col("contrib").sum().fill_null(0.0) / self.total_edges).cast(pl.Float64()))
            .item()
        )


@safe
def optimise_communities(adj_mat: AdjMat) -> AdjMat:
    connected_nodes = set(adj_mat.dwm.weighted_edges["src"]) | set(
        adj_mat.dwm.weighted_edges["dst"]
    )
    isolated = adj_mat.communities.filter(~pl.col("node").is_in(connected_nodes))
    adj_mat.communities = adj_mat.communities.filter(pl.col("node").is_in(connected_nodes))

    valid_merges = get_merge_scores(adj_mat)

    while not valid_merges.is_empty():
        to_merge = remove_overlapping_pairs(valid_merges)
        adj_mat.communities = apply_merges_lf(adj_mat.communities, to_merge)
        valid_merges = get_merge_scores(adj_mat)

    adj_mat.communities = adj_mat.communities.with_columns(
        pl.col("node").cast(pl.Int64), pl.col("module").cast(pl.Int64)
    )

    isolated = isolated.with_columns(pl.col("node").cast(pl.Int64), pl.col("module").cast(pl.Int64))
    adj_mat.communities = pl.concat([adj_mat.communities, isolated])

    print(f"Post-optimisation DWM: {adj_mat.dwm.calc(adj_mat.communities)}")  # noqa: T201

    return adj_mat


def get_merge_scores(adj_mat: AdjMat) -> pl.DataFrame:
    base_score = adj_mat.dwm.calc(communities=adj_mat.communities)

    unique_comms = adj_mat.communities.unique("module", maintain_order=True)["module"].to_list()
    merge_scores = []

    for i, c1 in enumerate(unique_comms):
        for c2 in unique_comms[i + 1 :]:
            score = adj_mat.dwm.calc(adj_mat.communities.with_columns(merge_communities(c1, c2)))
            gain = score - base_score
            merge_scores.append({"c1": c1, "c2": c2, "gain": gain})
    return pl.DataFrame(
        merge_scores, schema={"c1": pl.Int64(), "c2": pl.Int64(), "gain": pl.Float32()}
    ).filter(pl.col("gain") > 0)


def remove_overlapping_pairs(possible_pairs: pl.DataFrame) -> pl.DataFrame:
    selected, seen = [], set()

    for row in possible_pairs.sort("gain", descending=True).iter_rows(named=True):
        if row["c1"] not in seen and row["c2"] not in seen:
            selected.append(row)
            seen.add(row["c1"])
            seen.add(row["c2"])

    return pl.DataFrame(selected)


def apply_merges_lf(communities_lf: pl.DataFrame, merges_lf: pl.DataFrame) -> pl.DataFrame:
    mapping = merges_lf.select(pl.col("c2").alias("module"), pl.col("c1").alias("new_module"))
    return (
        communities_lf.join(mapping, on="module", how="left")
        .with_columns(pl.coalesce("new_module", "module").alias("module"))
        .drop("new_module")
    )


def merge_communities(community_1: int, community_2: int) -> pl.Expr:
    return (
        pl.when(pl.col("module") == community_2)
        .then(community_1)
        .otherwise(pl.col("module"))
        .alias("module")
    )
