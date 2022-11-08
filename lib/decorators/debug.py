import functools
from atexit import register
#from decorators import count_calls

def debug(func):
    """Print the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]                      # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)           # 3
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")           # 4
        return value
    return wrapper_debug


from dataclasses import dataclass

#
# @dataclass
# class Food:
#     name: str
#     unit_price: float
#     stock: int = 0
#
#     def stock_value(self) -> float:
#         return (self.stock * self.unit_price)

