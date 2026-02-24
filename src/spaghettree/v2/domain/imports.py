from __future__ import annotations

from enum import Enum

import attrs
from attrs.validators import instance_of


class ImportType(Enum):
    FROM = "from"
    IMPORT = "import"


@attrs.define(frozen=True)
class ImportCST:
    import_module: str = attrs.field(validator=[instance_of(str)])
    import_type: ImportType = attrs.field(validator=[instance_of(ImportType)])
    import_name: str = attrs.field(validator=[instance_of(str)])
    import_as_name: str = attrs.field(validator=[instance_of(str)])

    def to_str(self) -> str:
        output: list[str] = []
        if self.import_type is ImportType.FROM:
            output.append(f"from {self.import_module} import {self.import_name}")
        elif self.import_type is ImportType.IMPORT:
            output.append(f"import {self.import_module}")

        if self.import_name != self.import_as_name:
            output.append(f"as {self.import_as_name}")
        return " ".join(output)

    def to_dict(self) -> dict:
        res = attrs.asdict(self)
        res["import_type"] = res["import_type"].value
        return res
