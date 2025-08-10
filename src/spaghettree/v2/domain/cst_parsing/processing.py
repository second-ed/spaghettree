import os
from collections import defaultdict
from copy import deepcopy
from functools import partial

from spaghettree.v2 import safe
from spaghettree.v2.domain.cst_parsing.adj_mat import AdjMat
from spaghettree.v2.domain.cst_parsing.entities import ClassCST, FuncCST
from spaghettree.v2.domain.cst_parsing.globals import GlobalCST
from spaghettree.v2.domain.cst_parsing.imports import ImportCST
from spaghettree.v2.domain.cst_parsing.lib import (
    cst_to_str,
)


@safe
def create_new_module_map(
    adj_mat: AdjMat, entities: dict[str, FuncCST | ClassCST | GlobalCST]
) -> dict[int, list[FuncCST | ClassCST | GlobalCST]]:
    new_modules: defaultdict[int, list[FuncCST | ClassCST | GlobalCST]] = defaultdict(list)

    for i, module in enumerate(adj_mat.communities):
        ent_name = adj_mat.node_map[i]
        new_modules[module].append(entities[ent_name])

    return dict(new_modules)


@safe
def infer_module_names(
    new_modules: dict[int, list[FuncCST | ClassCST | GlobalCST]],
) -> dict[str, list[FuncCST | ClassCST | GlobalCST]]:
    renamed_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]] = {}

    for idx, contents in new_modules.items():
        if len(contents) > 1:
            names = [".".join(ent.name.split(".")[:-1]) for ent in new_modules[idx]]
            possible_module_names = sorted(
                set([(name, names.count(name)) for name in names]), key=lambda x: x[1], reverse=True
            )
            for name, _ in possible_module_names:
                if name not in renamed_modules:
                    mod_name = name
                    break
            else:
                mod_name = f"{possible_module_names[0][0]}.mod_overflow"
        else:
            mod_name = contents[0].name
        renamed_modules[mod_name] = contents
    return renamed_modules


@safe
def rename_overlapping_mod_names(
    renamed_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]],
) -> dict[str, list[FuncCST | ClassCST | GlobalCST]]:
    fixed_name_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]] = {}

    for name, contents in renamed_modules.items():
        name_parts = name.split(".")
        dirname = ".".join(name_parts[:-1])

        if dirname in renamed_modules:
            mod_name = ".".join([*name_parts[:-2], "_".join(name_parts[-2:])])
        else:
            mod_name = name

        fixed_name_modules[mod_name] = contents

    return fixed_name_modules


@safe
def create_new_filepaths(
    fixed_name_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]], src_root: str
) -> dict[str, list[FuncCST | ClassCST | GlobalCST]]:
    filepath_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]] = {}
    for name, contents in fixed_name_modules.items():
        new_name = os.path.join(os.path.dirname(src_root), name.replace(".", "/") + ".py")
        filepath_modules[new_name] = contents

    return filepath_modules


@safe
def convert_to_code_str(
    new_modules: dict[str, list[FuncCST | ClassCST | GlobalCST]], type_priority: dict[str, int]
) -> dict[str, str]:
    def get_module_str(mod_contents: list[FuncCST | ClassCST | GlobalCST]) -> str:
        imports, code = [], []

        for ent in mod_contents:
            imports.extend([imp.to_str() for imp in ent.imports])
            code.append(cst_to_str(ent.tree))

        return "".join(sorted(set(imports))) + "".join(code)

    def sort_by_priority(
        contents: list[FuncCST | ClassCST | GlobalCST], type_priority: dict[str, int]
    ) -> list[FuncCST | ClassCST | GlobalCST]:
        def sort_key(obj, type_priority: dict[str, int]):
            return (type_priority[obj.__class__.__name__], getattr(obj, "name", ""))

        return sorted(contents, key=partial(sort_key, type_priority=type_priority))

    return {k: get_module_str(sort_by_priority(v, type_priority)) for k, v in new_modules.items()}


@safe
def remap_imports(
    modules: dict[str, list[FuncCST | ClassCST | GlobalCST]],
) -> dict[str, list[FuncCST | ClassCST | GlobalCST]]:
    modules = deepcopy(modules)
    entity_mod_map: dict[str, str] = {
        ent.name: mod_name for mod_name, ents in modules.items() for ent in ents
    }

    for mod_name, ents in modules.items():
        for ent in ents:
            updated_imports: list[ImportCST] = []

            for imp in ent.imports:
                full_import = f"{imp.module}.{imp.name}"

                if full_import in entity_mod_map:
                    new_mod = entity_mod_map[full_import]

                    if new_mod == mod_name:
                        # same module after refactor: drop import
                        continue

                    updated_imports.append(
                        ImportCST(
                            module=new_mod,
                            import_type=imp.import_type,
                            name=imp.name,
                            as_name=imp.as_name,
                        )
                    )
                else:
                    # no changes
                    updated_imports.append(imp)

            ent.imports = updated_imports
    return modules
