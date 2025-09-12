import math as mt

CONSTANT: float = 1_000.99


def func_a() -> int:
    return mt.ceil(CONSTANT)


def func_b() -> int:
    return func_a() + func_a()
