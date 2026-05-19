from typing import Any

import attrs
import libcst as cst
import libcst.matchers as m
from libcst.metadata import FullRepoManager

from spaghettree.v2.domain.imports import ImportCST, ImportType

METADATA_DEPS = (
    cst.metadata.PositionProvider,
    cst.metadata.ScopeProvider,
    cst.metadata.FullyQualifiedNameProvider,
    cst.metadata.FilePathProvider,
)


class MetadataBase(cst.CSTVisitor):
    METADATA_DEPENDENCIES = METADATA_DEPS


@attrs.define(frozen=True)
class NodeMetadata:
    node: cst.CSTNode = attrs.field(repr=False)
    position: cst.metadata.CodeRange
    scope: cst.metadata.Scope
    qualified_names: tuple[cst.metadata.QualifiedName] = attrs.field(converter=[sorted, tuple])
    filepath: str
    calls: list = attrs.field(factory=list)
    references: list = attrs.field(factory=list)
    imports: list = attrs.field(factory=list)

    def to_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)


def get_manager(root: str, paths: list[str]) -> FullRepoManager:
    return FullRepoManager(root, paths=paths, providers=METADATA_DEPS)


@attrs.define
class NodeCollector(MetadataBase):
    entity_matchers: m.BaseMatcherNode
    reference_matchers: m.BaseMatcherNode
    import_matchers: m.BaseMatcherNode
    entities: list[NodeMetadata] = attrs.field(factory=list)
    imports: list = attrs.field(factory=list)
    filepath: str = attrs.field(default="")

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool | None:  # noqa: N802
        self._handle_entity(node)
        return super().visit_AnnAssign(node)

    def visit_Assign(self, node: cst.Assign) -> bool | None:  # noqa: N802
        self._handle_entity(node)
        return super().visit_Assign(node)

    def visit_Call(self, node: cst.Call) -> bool | None:  # noqa: N802
        if self.entities:
            self.entities[-1].calls.extend(self.get_metadata(cst.metadata.FullyQualifiedNameProvider, node, set()))
        return super().visit_Call(node)

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:  # noqa: N802
        self._handle_entity(node)
        return super().visit_ClassDef(node)

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:  # noqa: N802
        self._handle_entity(node)
        return super().visit_FunctionDef(node)

    def visit_Import(self, node: cst.Import) -> bool | None:  # noqa: N802
        self._handle_import(node)
        return super().visit_Import(node)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:  # noqa: N802
        self._handle_import(node)
        return super().visit_ImportFrom(node)

    def visit_Name(self, node: cst.Name) -> bool | None:  # noqa: N802
        if self.entities:
            name = self.get_metadata(cst.metadata.FullyQualifiedNameProvider, node, set())

            if not self.entities[-1].qualified_names:
                object.__setattr__(self.entities[-1], "qualified_names", tuple(name))

            self.entities[-1].references.extend(name)
        return super().visit_Name(node)

    def _handle_entity(self, node: cst.CSTNode) -> None:
        if not m.matches(node, self.entity_matchers):
            return

        scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)

        if not isinstance(scope, cst.metadata.scope_provider.GlobalScope):
            return

        self.entities.append(
            NodeMetadata(
                node=node,
                position=self.get_metadata(cst.metadata.PositionProvider, node, None),
                scope=self.get_metadata(cst.metadata.ScopeProvider, node, None),
                qualified_names=self.get_metadata(cst.metadata.FullyQualifiedNameProvider, node, set()),
                filepath=self.filepath,
            )
        )

    def _handle_import(self, node: cst.CSTNode) -> None:
        if not m.matches(node, self.import_matchers):
            return
        if m.matches(node, m.Import()):
            self._record_import(None, node.names, ImportType.IMPORT)
            return

        module = self._resolve_attr(node.module)
        if module is None:
            # skip relative imports
            return

        if m.matches(node, m.ImportFrom()):
            if m.matches(node.names, m.ImportStar()):
                self.imports.append(ImportCST(module, ImportType.FROM, "*", "*"))
                return

            aliases = [node.names] if isinstance(node.names, cst.ImportAlias) else node.names
            self._record_import(module, aliases, ImportType.FROM)

    def _record_import(self, module: str | None, aliases: list, import_type: ImportType) -> None:
        for alias in aliases:
            name = self._resolve_attr(alias.name)
            as_name = alias.asname.name.value if alias.asname else name
            self.imports.append(ImportCST(module or name, import_type, name, as_name))

    def _resolve_attr(self, node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parent = self._resolve_attr(node.value)
            return f"{parent}.{node.attr.value}" if parent else node.attr.value
        if isinstance(node, cst.Subscript):
            return self._resolve_attr(node.value)
        return None
