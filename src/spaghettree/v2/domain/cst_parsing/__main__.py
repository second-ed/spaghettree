from functools import partial

from spaghettree.v2 import Result
from spaghettree.v2.adapters.io_wrapper import IOWrapper
from spaghettree.v2.domain.cst_parsing.adj_mat import AdjMat
from spaghettree.v2.domain.cst_parsing.lib import (
    create_call_tree,
    create_module_cst_objs,
    extract_entities,
    filter_non_native_calls,
    pair_exclusive_calls,
    resolve_module_calls,
)
from spaghettree.v2.domain.cst_parsing.optimisation import optimise_communities
from spaghettree.v2.domain.cst_parsing.processing import (
    convert_to_code_str,
    create_new_filepaths,
    create_new_module_map,
    infer_module_names,
    remap_imports,
    rename_overlapping_mod_names,
)


def main(src_root: str, new_root: str) -> Result:
    io = IOWrapper()
    entities_res = (
        io.read_files(src_root)
        .bind(create_module_cst_objs)
        .bind(resolve_module_calls)
        .bind(extract_entities)
        .bind(filter_non_native_calls)
    )

    entities = entities_res.inner

    return (
        entities_res.bind(create_call_tree)
        .bind(AdjMat.from_call_tree)
        .bind(pair_exclusive_calls)
        .bind(optimise_communities)
        .bind(partial(create_new_module_map, entities=entities))
        .bind(infer_module_names)
        .bind(rename_overlapping_mod_names)
        .bind(remap_imports)
        .bind(convert_to_code_str)
        .bind(partial(create_new_filepaths, src_root=new_root))
        .bind(partial(io.write_files, ruff_root=new_root))
    )
