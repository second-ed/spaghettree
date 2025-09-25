import argparse
import json
from pathlib import Path

from spaghettree import Result
from spaghettree.adapters.io_wrapper import IOProtocol, IOWrapper
from spaghettree.domain.optimisation import (
    AdjMat,
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


def main(
    src_root: str,
    *,
    new_root: str = "",
    call_tree_save_path: str = "./call_tree.json",
    optimise_src_code: bool = False,
) -> Result:
    io = IOWrapper()
    return run_process(
        io,
        src_root,
        new_root=new_root,
        optimise_src_code=optimise_src_code,
        call_tree_save_path=call_tree_save_path,
    )


def run_process(
    io: IOProtocol,
    src_root: str,
    *,
    new_root: str = "",
    call_tree_save_path: str = "./call_tree.json",
    optimise_src_code: bool = False,
) -> Result:
    logger.info(f"*** RUNNING `spaghettree` {src_root = } {new_root = } ***")
    src_code = io.read_files(src_root).unwrap()

    ent_and_locs_res = extract_entities_and_locations(src_code)
    entities, location_map = ent_and_locs_res.unwrap()
    entities_res = filter_non_native_calls(entities)
    entities = entities_res.unwrap()
    call_tree = entities_res.and_then(create_call_tree).unwrap()

    if optimise_src_code:
        res = optimise_entity_positions(
            entities=entities,
            location_map=location_map,
            call_tree=call_tree,
            src_root=src_root,
            new_root=new_root,
        ).unwrap()
    else:
        # remove any new_root so that it doesn't try to use ruff on the json
        new_root = ""
        adj_mat = AdjMat.from_call_tree(call_tree, optimise=optimise_src_code).unwrap()
        print(  # noqa: T201
            yellow(
                f"Current Directed Weighted Modularity (DWM): {get_dwm(adj_mat.mat, adj_mat.communities): .5f}"
            )
        )
        top_merges = get_top_suggested_merges(adj_mat).unwrap()

        for merge in top_merges:
            merge.display()

        res = {Path(call_tree_save_path).absolute(): json.dumps(call_tree, indent=4)}

    return io.write_files(res, ruff_root=new_root, format_bulk=optimise_src_code)


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
        "--call-tree-save-path",
        dest="call_tree_save_path",
        type=str,
        default="./call_tree.json",
        help="The location to save the generated call tree. Only used if `--optimise-src-code` isn't used. Defaults to `./call_tree.json`.",
    )
    parser.add_argument(
        "--optimise-src-code",
        dest="optimise_src_code",
        action="store_true",
        help="Enable optimisation of the source code.",
    )

    args = parser.parse_args()
    res = main(
        args.src_root,
        new_root=args.new_root,
        call_tree_save_path=args.call_tree_save_path,
        optimise_src_code=args.optimise_src_code,
    )
    if not res.is_ok():
        raise res.error
