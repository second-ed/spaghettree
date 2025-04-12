from __future__ import annotations

from returns.maybe import Maybe, Nothing, Some
from returns.result import Failure, Result, Success

from spaghettree.utils import format_code_str


async def get_src_code(path: str) -> Maybe[str]:
    try:
        with open(path, "r") as f:
            src_code = f.read()
        return Some(src_code)
    except Exception as e:
        print(f"{e} for {path}")
        return Nothing


async def save_modified_code(
    modified_code: str, filepath: str, format_code: bool = True
) -> Result[bool, Exception]:
    try:
        if format_code:
            modified_code = format_code_str(modified_code)
        with open(filepath, "w") as f:
            f.write(modified_code)
        return Success(True)
    except Exception as e:
        print(f"{e} for {filepath}")
        return Failure(e)
