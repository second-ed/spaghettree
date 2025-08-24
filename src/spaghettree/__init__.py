from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
)

import attrs

if TYPE_CHECKING:
    from types import TracebackType


def make_type(
    name: str,
    validators: Callable | Sequence[Callable] | None = None,
    converters: Callable | Sequence[Callable] | None = None,
    *,
    frozen: bool = True,
    eq: bool = True,
) -> type:
    def to_list(value: Callable | Sequence[Callable]) -> list:
        return list(value) if isinstance(value, Sequence) else [value] if value else []  # type: ignore[truthy-function]

    kwargs = {}

    if validators:
        kwargs["validator"] = to_list(validators)
    if converters:
        kwargs["converter"] = to_list(converters)

    return attrs.define(frozen=frozen, eq=eq)(
        type(
            name,
            (),
            {
                "inner": attrs.field(**kwargs),  # type: ignore[call-overload]
                "new": classmethod(lambda cls, val: cls(val)),
                "clone": lambda self: type(self)(self.inner),
                "map": lambda self, fn: type(self)(fn(self.inner)),
                "apply": lambda self, fn: fn(self.inner),
            },
        ),
    )


T = TypeVar("T")
U = TypeVar("U")
P = ParamSpec("P")


@attrs.define
class Ok:
    inner: Any = attrs.field(default=None)

    def is_ok(self) -> Literal[True]:
        return True

    def and_then(self, func: Callable[[T], Result]) -> Result:
        return func(self.inner)


@attrs.define
class Err:
    input_args: Any = attrs.field(repr=False)
    error: Exception | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(Exception)),
    )
    err_type: BaseException = attrs.field(init=False, repr=False)
    err_msg: str = attrs.field(init=False, repr=False)
    details: list[dict[str, Any]] = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.err_type = type(self.error)
        self.err_msg = str(self.error)
        if self.error:
            self.details = self._extract_details(self.error.__traceback__)

    def _extract_details(self, tb: TracebackType | None) -> list[dict[str, Any]]:
        trace_info = []
        while tb:
            frame = tb.tb_frame
            trace_info.append(
                {
                    "file": frame.f_code.co_filename,
                    "func": frame.f_code.co_name,
                    "line_no": tb.tb_lineno,
                    "locals": frame.f_locals,
                },
            )
            tb = tb.tb_next
        return trace_info

    def is_ok(self) -> Literal[False]:
        return False

    def and_then(self, _: Callable[[T], Result]) -> Self:
        return self


Result = Ok | Err


def safe[**P, T](func: Callable[P, T]) -> Callable[P, Result]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            return Err((args, kwargs), e)

    return wrapper


__all__ = [
    "Err",
    "Ok",
    "Result",
]
