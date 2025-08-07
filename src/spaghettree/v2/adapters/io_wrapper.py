import attrs
import black
import isort

from spaghettree.v2 import safe


@attrs.define
class IOWrapper:
    @safe
    def read(self, path: str) -> str:
        with open(path, "r") as f:
            data = f.read()
        return data

    @safe
    def write(self, modified_code: str, filepath: str, format_code: bool = True) -> None:
        def format_code_str(code_snippet: str) -> str:
            return black.format_str(isort.code(code_snippet), mode=black.FileMode())

        if format_code:
            modified_code = format_code_str(modified_code)
        with open(filepath, "w") as f:
            f.write(modified_code)
