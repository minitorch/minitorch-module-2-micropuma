"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    return a * b


def id(a: float) -> float:
    return a


def add(a: float, b: float) -> float:
    return a + b


def neg(a: float) -> float:
    return float(-a)


def lt(a: float, b: float) -> float:
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    return a if a >= b else b


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> float:
    return abs(a - b) < 1e-5


def sigmoid(x: float) -> float:
    """
    Implement sigmoid operation, basic math function is 1/1+e^(-x)
    Args:
        x: a float input
    Returns:
        sigmoid function applied on x input
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))  # 数值稳定版
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    assert x != 0
    return (1.0 / x) * d


def inv_back(x: float, d: float) -> float:
    assert x != 0
    return (-1.0 / (x**2)) * d


def relu_back(x: float, d: float) -> float:
    return d if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], a: Iterable[float]) -> Iterable[float]:
    """ "
    Higher-order function that applies a given function to each element of an iterable
    """
    return [fn(a_) for a_ in a]


def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """
    Higher-order function that combines elements from two iterables using a given function
    """
    return [fn(a, b) for a, b in zip(a, b)]


def reduce(
    fn: Callable[[float, float], float], a: Iterable[float], init: float
) -> Callable[[Iterable[float]], float]:
    def reduceImpl(aImpl: Iterable[float]) -> float:
        result = init
        for a_ in a:
            result = fn(result, a_)
        return result

    return reduceImpl(a)


def negList(a: Iterable[float]) -> Iterable[float]:
    """
    Neg the whole list, with the help of map function
    """
    return map(neg, a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """
    Add corresponding elements from two lists using zipWith
    """
    return zipWith(add, a, b)


def sum(a: Iterable[float]) -> float:
    """
    Sum all elements in a list using reduce
    """
    return reduce(add, a, 0.0)


def prod(a: Iterable[float]) -> float:
    """
    Calculate the product of all elements in a list using reduce
    """
    return reduce(mul, a, 1.0)
