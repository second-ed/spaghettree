import argparse
from pprint import pformat

from spaghettree import Ok, Result
from spaghettree.adapters.io_wrapper import IOProtocol, IOWrapper
from spaghettree.domain.adj_mat import AdjMat
from spaghettree.domain.optimisation import (
    cyan,
    get_dwm,
    get_top_suggested_merges,
    yellow,
)
from spaghettree.domain.parsing import (
    create_call_tree,
    extract_entities_and_locations,
    filter_non_native_calls,
)
from spaghettree.domain.processing import (
    optimise_entity_positions,
)
from spaghettree.logger import logger


def main(src_root: str, *, new_root: str = "", optimise_src_code: bool = False) -> Result:
    io = IOWrapper()
    return run_process(io, src_root, new_root=new_root, optimise_src_code=optimise_src_code)


def run_process(
    io: IOProtocol, src_root: str, *, new_root: str = "", optimise_src_code: bool = False
) -> Result:
    logger.info(f"*** RUNNING `spaghettree` {src_root = } {new_root = } ***")
    src_code = io.read_files(src_root).unwrap()

    ent_and_locs_res = extract_entities_and_locations(src_code)
    entities, location_map = ent_and_locs_res.unwrap()
    entities_res = filter_non_native_calls(entities)
    entities = entities_res.unwrap()
    call_tree = entities_res.and_then(create_call_tree).unwrap()

    if optimise_src_code:
        return optimise_entity_positions(
            io=io,
            entities=entities,
            location_map=location_map,
            call_tree=call_tree,
            src_root=src_root,
            new_root=new_root,
        )

    adj_mat = AdjMat.from_call_tree_no_optimisation(call_tree).unwrap()
    print(  # noqa: T201
        yellow(
            f"Current Directed Weighted Modularity (DWM): {get_dwm(adj_mat.mat, adj_mat.communities): .5f}"
        )
    )
    top_merges = get_top_suggested_merges(adj_mat).unwrap()

    for merge in top_merges:
        merge.display()

    return Ok(call_tree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process source code from a given root, with optional relocation and optimisation."
    )

    parser.add_argument("src_root", type=str, help="Path to the source root directory.")
    parser.add_argument(
        "--new-root",
        dest="new_root",
        type=str,
        default="",
        help="Optional new root path for output (default: empty, meaning same as src_root if optimisation is enabled).",
    )
    parser.add_argument(
        "--optimise-src-code",
        dest="optimise_src_code",
        action="store_true",
        help="Enable optimisation of the source code.",
    )

    args = parser.parse_args()
    res = main(args.src_root, new_root=args.new_root, optimise_src_code=args.optimise_src_code)
    call_tree = res.unwrap()
    print(f"\n{cyan(pformat(call_tree))}")  # noqa: T201
