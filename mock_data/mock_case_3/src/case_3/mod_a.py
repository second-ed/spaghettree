import math  # noqa: INP001


class A:
    pass


def func_a() -> int:
    return math.ceil(0.5)


def func_b() -> int:
    return func_a() + func_a()
