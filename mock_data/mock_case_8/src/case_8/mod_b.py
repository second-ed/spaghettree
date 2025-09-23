from case_8.logger import logger
from case_8.mod_a import func_a


def func_b() -> int:
    logger.debug("calling func b")
    return func_a() + float(func_a())
