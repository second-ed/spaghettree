import attrs
import libcst as cst
import libcst.matchers as m
from libcst.metadata import FullRepoManager

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
    qualified_names: tuple[cst.metadata.QualifiedName] = attrs.field(converter=tuple)
    filepath: str
    calls: list = attrs.field(factory=list)
    references: list = attrs.field(factory=list)


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

    def on_visit(self, node: cst.CSTNode) -> bool:
        if m.matches(node, self.entity_matchers):
            scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)

            if not isinstance(scope, cst.metadata.scope_provider.GlobalScope):
                return super().on_visit(node)

            self.entities.append(
                NodeMetadata(
                    node=node,
                    position=self.get_metadata(cst.metadata.PositionProvider, node, None),
                    scope=self.get_metadata(cst.metadata.ScopeProvider, node, None),
                    qualified_names=self.get_metadata(
                        cst.metadata.FullyQualifiedNameProvider, node, set()
                    ),
                    filepath=self.filepath,
                )
            )
            return super().on_visit(node)

        if m.matches(node, self.import_matchers):
            self.imports.append(
                NodeMetadata(
                    node=node,
                    position=self.get_metadata(cst.metadata.PositionProvider, node, None),
                    scope=self.get_metadata(cst.metadata.ScopeProvider, node, None),
                    qualified_names=self.get_metadata(
                        cst.metadata.FullyQualifiedNameProvider, node, set()
                    ),
                    filepath=self.filepath,
                )
            )
            return super().on_visit(node)

        if not self.entities:
            return super().on_visit(node)

        if m.matches(node, m.Call()):
            self.entities[-1].calls.extend(
                self.get_metadata(cst.metadata.FullyQualifiedNameProvider, node, set())
            )

        if m.matches(node, m.Name()):
            name = self.get_metadata(cst.metadata.FullyQualifiedNameProvider, node, set())

            if not self.entities[-1].qualified_names:
                object.__setattr__(self.entities[-1], "qualified_names", tuple(name))

            self.entities[-1].references.extend(name)

        return super().on_visit(node)
