import time
from functools import wraps


def timefn(fn):
    """The execution time of function."""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        return result, total_time

    return measure_time
