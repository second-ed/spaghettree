from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

import attrs
import black
import isort
from ruff.__main__ import find_ruff_bin

from spaghettree import Err, Ok, Result, safe


@runtime_checkable
class IOProtocol(Protocol):
    @safe
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]: ...

    @safe
    def read(self, path: str) -> str: ...

    @safe
    def read_files(self, root: str | Path) -> Result: ...

    @safe
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None: ...

    def write_files(self, src_code: dict[str, str], ruff_root: str | None = None) -> Result: ...


@attrs.define
class IOWrapper:
    @safe
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]:
        return glob.glob(f"{root}/**/**.py", recursive=recursive)

    @safe
    def read(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def read_files(self, root: str | Path) -> Result:
        paths_res = self.list_files(root)
        if not paths_res.is_ok():
            return paths_res
        paths = paths_res.inner

        results, fails = {}, {}
        for path in paths:
            res = self.read(path)
            if res.is_ok():
                results[path] = res.inner
            else:
                fails[path] = res

        if fails:
            return Err(fails)
        return Ok(results)

    @safe
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(format_code_str(modified_code))
        if format_code:
            self._run_ruff(filepath)

    def write_files(self, src_code: dict[str, str], ruff_root: str | None = None) -> Result:
        results, fails = {}, {}

        for filepath, modified_code in src_code.items():
            if ruff_root is not None:
                # format all at the end instead
                res = self.write(modified_code, filepath, format_code=False)
            else:
                res = self.write(modified_code, filepath, format_code=True)

            if res.is_ok():
                results[filepath] = res.inner
            else:
                fails[filepath] = res

        if ruff_root:
            self._run_ruff(ruff_root)
        if fails:
            return Err(fails)
        return Ok(results)

    def _run_ruff(self, path: str) -> None:
        subprocess.run([find_ruff_bin(), "check", "--fix", str(path)], check=True)  # noqa: S603
        subprocess.run([find_ruff_bin(), "format", str(path)], check=True)  # noqa: S603


@attrs.define
class FakeIOWrapper:
    files: dict = attrs.field(factory=dict)

    @safe
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]:
        if recursive:
            return [f for f in self.files if f.startswith(root) and f.endswith(".py")]
        return [
            f for f in self.files if f.removeprefix(root).lstrip("/").split("/")[0].endswith(".py")
        ]

    @safe
    def read(self, path: str) -> str:
        return self.files[path]

    def read_files(self, root: str | Path) -> Result:
        paths_res = self.list_files(root)
        if not paths_res.is_ok():
            return paths_res
        paths = paths_res.inner

        results, fails = {}, {}
        for path in paths:
            res = self.read(path)
            if res.is_ok():
                results[path] = res.inner
            else:
                fails[path] = res

        if fails:
            return Err(fails)
        return Ok(results)

    @safe
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None:
        self.files[filepath] = format_code_str(modified_code) if format_code else modified_code

    def write_files(self, src_code: dict[str, str], ruff_root: str | None = None) -> Result:
        results, fails = {}, {}

        for filepath, modified_code in src_code.items():
            if ruff_root is not None:
                res = self.write(modified_code, filepath)

            if res.is_ok():
                results[filepath] = res.inner
            else:
                fails[filepath] = res

        if fails:
            return Err(fails)
        return Ok(results)


def format_code_str(code: str) -> str:
    return black.format_str(isort.code(code), mode=black.FileMode())
