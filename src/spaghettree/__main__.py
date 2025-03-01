import os
from copy import deepcopy
from typing import Optional

import attrs
import black
import isort
import libcst as cst
from attrs.validators import instance_of


def format_code_str(code_snippet: str) -> str:
    return black.format_str(isort.code(code_snippet), mode=black.FileMode())


def get_src_code(path: str) -> str:
    with open(path, "r") as f:
        src_code = f.read()
    return src_code


def save_modified_code(
    modified_code: str, filepath: str, format_code: bool = True
) -> bool:
    try:
        if format_code:
            modified_code = format_code_str(modified_code)
        with open(filepath, "w") as f:
            f.write(modified_code)
        return True
    except Exception as e:
        print(f"{e} for {filepath}")
        return False


def remove_duplicate_calls(calls: list[str]) -> list:
    return list(dict.fromkeys(calls))


def str_to_cst(code: str) -> cst.Module:
    return cst.parse_module(code)


def cst_to_str(node) -> str:
    return cst.Module([]).code_for_node(node)


def collect_names(json):
    def _collect_names(node, prefix=""):
        name = node.get("name", "")
        if name:
            if prefix:
                name = ".".join([prefix, name])
            names.append(name)
        for n in node.get("methods", []):
            _collect_names(n, name)

    names = []

    for node in json.values():
        _collect_names(node)

    return names


def filter_tree(json, defined_names):
    json = deepcopy(json)

    def _filter_calls(node):
        if "calls" in node:
            node["calls"] = [c for c in node.get("calls", []) if c in defined_names]

        for m in node.get("methods", []):
            _filter_calls(m)

    for node in json.values():
        _filter_calls(node)

    return json


def get_module_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _get_full_module_name(module) -> Optional[str]:
    if isinstance(module, cst.Attribute):
        return (
            _get_full_module_name(module.value)  # type: ignore
            + "."
            + module.attr.value
        )
    elif isinstance(module, cst.Name):
        return module.value
    return None


@attrs.define
class ModuleCST:
    name: str = attrs.field(validator=[])
    tree: cst.Module = attrs.field(validator=[], repr=False)
    imports: list = attrs.field(default=None, repr=False)
    func_trees: dict = attrs.field(default=None, repr=False)
    class_trees: dict = attrs.field(default=None, repr=False)

    def __attrs_post_init__(self):
        self.imports = [
            node
            for node in self.tree.children
            if isinstance(node, cst.SimpleStatementLine)
            and isinstance(node.body[0], (cst.ImportFrom, cst.Import))
        ]
        self.func_trees = {
            node.name.value: node
            for node in self.tree.children
            if isinstance(node, cst.FunctionDef)
        }
        self.class_trees = {
            node.name.value: node
            for node in self.tree.children
            if isinstance(node, cst.ClassDef)
        }


@attrs.define
class ClassCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.ClassDef = attrs.field(validator=[instance_of(cst.ClassDef)], repr=False)
    methods: list = attrs.field(validator=[instance_of(list)])
    module_name: str = attrs.field(validator=[instance_of(str)])


@attrs.define
class FuncCST:
    name: str = attrs.field(validator=[instance_of(str)])
    tree: cst.FunctionDef = attrs.field(
        validator=[instance_of(cst.FunctionDef)], repr=False
    )
    calls: list = attrs.field(validator=[instance_of(list)])
    indent: int = attrs.field(validator=[instance_of(int)])
    module_name: str = attrs.field(validator=[instance_of(str)])
    class_name: str = attrs.field(default="", validator=[instance_of(str)])
    internal: bool = attrs.field(default=False, validator=[instance_of(bool)])


@attrs.define
class CallVisitor(cst.CSTVisitor):
    depth: int = attrs.field(default=0, validator=[instance_of(int)])  # type: ignore
    calls: list = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.calls = []

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth += 1

    def leave_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth -= 1

    def visit_Call(self, node: cst.Call) -> None:
        if isinstance(node.func, cst.Name):
            self.calls.append(node.func.value)
        elif isinstance(node.func, cst.Attribute):
            if isinstance(node.func.value, cst.Name):
                self.calls.append(f"{node.func.value.value}.{node.func.attr.value}")


@attrs.define
class FuncVisitor(cst.CSTVisitor):
    module_name: str = attrs.field(default="", validator=[instance_of(str)])
    class_name: str = attrs.field(default="", validator=[instance_of(str)])
    depth: int = attrs.field(default=0, validator=[instance_of(int)])  # type: ignore
    funcs: list = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.funcs = []

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth += 1

    def leave_IndentedBlock(self, node: cst.IndentedBlock) -> None:
        self.depth -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        cv = CallVisitor()
        node.visit(cv)
        self.funcs.append(
            FuncCST(
                node.name.value,
                node,
                cv.calls,
                cv.depth,
                self.module_name,
                self.class_name,
            )
        )
