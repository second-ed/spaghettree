class A:  # noqa: INP001
    pass


def func_a() -> int:
    return 0


def func_b() -> int:
    return func_a() + func_a()
