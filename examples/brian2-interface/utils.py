from functools import wraps
import time
import numpy as np

_profilerDict = {}


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        if func.__name__ not in _profilerDict:
            _profilerDict[func.__name__] = []
        _profilerDict[func.__name__].append(total_time)
        # print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def print_time_summary():
    sortedDict = dict(sorted(_profilerDict.items()))
    window_size = 3
    for key in sortedDict:
        avg = (
            0
            if len(_profilerDict[key]) <= window_size
            else np.array(_profilerDict[key])[:-window_size].mean()
        )
        print(f"function: {key} \t\t{avg:.4f} sec")


def flush_and_print_time_summary():
    print_time_summary()
    _profilerDict = {}
