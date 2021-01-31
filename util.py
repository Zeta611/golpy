import time
from functools import wraps
from typing import Dict


class timeit:
    records: Dict[str, float] = {}
    on = False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.on:
                return func(*args, **kwargs)

            name = func.__name__

            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            elapsed = end - start
            if name in self.records:
                self.records[name] += elapsed
            else:
                self.records[name] = elapsed
            return result

        return wrapper
