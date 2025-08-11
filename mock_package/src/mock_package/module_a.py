from mock_package.module_b import func_d


def func_a():
    return 1 + func_b()


def func_b():
    return 1


def func_c():
    return 2 * func_d()


def isolated_func():
    return 2
