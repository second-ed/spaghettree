from case_1.mod_a import func_a


def func_b() -> int:
    return func_a() + float(func_a())
