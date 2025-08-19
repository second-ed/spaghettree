from mock_package.module_b import func_d  # noqa: INP001


def func_a():  # noqa: ANN201
    return 1 + func_b()


def func_b():  # noqa: ANN201
    return 1


def func_c():  # noqa: ANN201
    return 2 * func_d()


def isolated_func():  # noqa: ANN201
    return 2
