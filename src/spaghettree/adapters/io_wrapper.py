from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

import attrs
import black
import isort
from danom import Err, Ok, Result, safe
from ruff.__main__ import find_ruff_bin

from spaghettree.core.logger import logger
from spaghettree.domain.optimisation import yellow


@attrs.define
class IOBase(ABC):
    src_dirname: str = attrs.field(
        default="src", converter=lambda x: str(x).strip("/"), validator=attrs.validators.instance_of(str)
    )
    src_files: dict[str, str] = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    tests_dirname: str = attrs.field(
        default="tests", converter=lambda x: str(x).strip("/"), validator=attrs.validators.instance_of(str)
    )
    test_files: dict[str, str] = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    ignore_dirs: list[str] = attrs.field(factory=list, validator=attrs.validators.instance_of(list))

    @abstractmethod
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]:
        pass

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    @abstractmethod
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None:
        pass

    @abstractmethod
    def _run_ruff(self, path: str) -> None:
        pass

    def read_files(self, root: str | Path) -> Result[dict[str, str], dict[str, Err]]:
        paths_res = self.list_files(root)
        if not paths_res.is_ok():
            return paths_res
        paths = paths_res.inner

        logger.debug(f"{paths = }")

        fails = {}
        for path in paths:
            if any(ignore_dir in path for ignore_dir in self.ignore_dirs):
                continue
            res = self.read(path)
            if res.is_ok():
                if f"/{self.src_dirname}/" in path:
                    self.src_files[path] = res.inner
                elif f"/{self.tests_dirname}/" in path and (
                    Path(path).stem.startswith("test_") or Path(path).stem in ["__init__", "conftest"]
                ):
                    self.test_files[path] = res.inner
            else:
                fails[path] = res

        logger.debug(f"{self.src_files = }")
        logger.debug(f"{self.test_files = }")

        if fails:
            logger.error(f"{fails = }")
            return Err(fails)
        return Ok(self.src_files)

    def write_files(
        self, src_code: dict[str, str], ruff_root: str | None = None, *, format_bulk: bool = True
    ) -> Result:
        results, fails = {}, {}

        format_code = ruff_root and not format_bulk

        for filepath, modified_code in src_code.items():
            res: Result = self.write(modified_code, filepath, format_code=format_code)

            logger.debug(f"{filepath = } {res = }")
            if res.is_ok():
                print(yellow(f"File written to `{filepath}`"))  # noqa: T201
                results[filepath] = res.inner
            else:
                logger.error(yellow(f"failed to write {filepath = } {res.err_msg = }"))
                fails[filepath] = res

        if ruff_root and format_bulk:
            self._run_ruff(ruff_root)
        if fails:
            return Err(fails)
        return Ok(results)


@attrs.define
class IOWrapper(IOBase):
    @safe
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]:
        root_path = Path(root).resolve()
        files = root_path.rglob("*.py") if recursive else root_path.glob("*.py")
        return sorted(str(f) for f in files if f.is_file())

    @safe
    def read(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    @safe
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(modified_code, encoding="utf-8")
        if format_code:
            self._run_ruff(filepath)

    def _run_ruff(self, path: str) -> None:
        subprocess.run([find_ruff_bin(), "check", "--fix", str(path)], check=True)  # noqa: S603
        subprocess.run([find_ruff_bin(), "format", str(path)], check=True)  # noqa: S603


@attrs.define
class FakeIOWrapper(IOBase):
    files: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))

    @safe
    def list_files(self, root: str | Path, *, recursive: bool = True) -> list[str]:
        logger.debug(f"{self.files = }")
        if recursive:
            return sorted([f for f in self.files if root in f and f.endswith(".py")])
        return sorted([f for f in self.files if f.removeprefix(root).lstrip("/").split("/")[0].endswith(".py")])

    @safe
    def read(self, path: str) -> str:
        return self.files[path]

    @safe
    def write(self, modified_code: str, filepath: str, *, format_code: bool = True) -> None:
        self.files[filepath] = format_code_str(modified_code) if format_code else modified_code

    def _run_ruff(self, path: str) -> None:
        tgt_files = self.list_files(path).unwrap()
        self.files = {p: format_code_str(self.files[p]) if p.endswith(".py") else self.files[p] for p in tgt_files}


def format_code_str(code: str) -> str:
    return black.format_str(isort.code(code), mode=black.FileMode())
