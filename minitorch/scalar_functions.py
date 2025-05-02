from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch
import math

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike

# 构建tuple
def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    # 静态方法，也是variable tracking的核心函数
    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []

        # minitorch.scalar是对于每个variable的封装
        # 将每个value变成Scalar封装
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        # 解包可迭代对象为单独参数
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        # 构建这个节点的执行历史，包括该节点的context，输入scalars
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return wrap_tuple(operators.log_back(a, d_output))


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        # both a and b should be recorded in context
        # c = a x b
        # now we get ay/ac, and we want ay/aa
        # a possible we is ay/ac x ac/aa
        # ac/aa = b, so b should be kept
        # the same for a
        ctx.save_for_backward(a,b)
        return operators.mul(a,b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        (a,b) = ctx.saved_values
        return b * d_output ,a * d_output


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        try:
            assert(a != 0.0)
            ctx.save_for_backward(a)
            return operators.inv(a)
        except:
            raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return wrap_tuple(operators.inv_back(a, d_output))


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        return wrap_tuple(-d_output)


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        # store 1/(1+e^-a)
        s = 1.0 / (1.0 + math.exp(-a))
        ctx.save_for_backward(a)
        return s

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        s = 1.0 / (1.0 + math.exp(-a))
        return wrap_tuple(s * (1-s) * d_output)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return max(a, 0.0)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return wrap_tuple(operators.relu_back(a, d_output))


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        exp = math.exp(a)
        ctx.save_for_backward(a)
        return exp

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return wrap_tuple(d_output * operators.exp(a))


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        # not differentiable
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return (0.0, 0.0) 


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # TODO: Implement for Task 1.4.
        return (0.0, 0.0)
