from case_2.mod_b import func_d


def func_a() -> int:
    return 0 + func_b()


def func_b() -> int:
    return 1


def func_c() -> int:
    return 2 + func_d()


def isolated_func() -> int:
    return 5
