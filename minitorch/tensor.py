"""
Implementation of the core Tensor object for autodifferentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]

# 和scalar的定义一样
@dataclass
class History:
    """
    `History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


# tensor的逻辑实现
# 在tensor_data中是tensor的底层数据存储
class Tensor:
    """
    Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]

    # 底层tensor存储
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    # 根据是否有history历史信息，来判断是否需要求导
    def requires_grad_(self, x: bool) -> None:
        self.history = History()

    def requires_grad(self) -> bool:
        return self.history is not None

    # 将tensor的封装unwrap，直接暴露底层的存储
    def to_numpy(self) -> npt.NDArray[np.float64]:
        """
        Returns:
             Converted to numpy array
        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    # Properties
    # 均需要访问底层的存储类型：tensorData，里面存储诸如storage，shape等信息
    @property
    def shape(self) -> UserShape:
        """
        Returns:
             shape of the tensor
        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """
        Returns:
             int : size of the tensor
        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """
        Returns:
             int : dimensionality of the tensor
        """
        return self._tensor.dims

    # 类型统一化工具，将所有参与运算的operand变成对应设备上的tensor
    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        "Turns a python number into a tensor with the same backend."
        # 针对单一元素，构建tensor object
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        # 单纯设定设备后端即可
        # todo: 不知道为什么这里会报错，所以打一个补丁
        elif isinstance(b, np.generic):  # 处理 NumPy 标量类型
            return Tensor.make([b.item()], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    # Functions
    # 类似scalar的设计实现，只是需要ensure_tensor这一个步骤
    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, self._ensure_tensor(b))

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        "Not used until Module 3"
        return MatMul.apply(self, b)

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        return ReLU.apply(self)

    def log(self) -> Tensor:
        return Log.apply(self)

    def exp(self) -> Tensor:
        return Exp.apply(self)

    def item(self) -> float:
        assert self.size == 1
        return self[0]

    # 一些tensor特色函数实现
    def sum(self, dim: Optional[int] = None) -> Tensor:
        "Compute the sum over dimension `dim`"
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        "Compute the mean over dimension `dim`"
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        "Permute tensor dimensions to *order"
        return Permute.apply(self, tensor(list(order)))

    # tensor view，提供视图
    def view(self, *shape: int) -> Tensor:
        "Change the shape of the tensor to a new shape with the same size"
        return View.apply(self, tensor(list(shape)))

    def contiguous(self) -> Tensor:
        "Return a contiguous tensor with the same data"
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        "Create a new tensor from data"
        # 理解一个tensor是如何构建的
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """
        Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Parameters:
            other : backward tensor (must broadcast with self)

        Returns:
            Expanded version of `other` with the right derivatives

        """

        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x : value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0] * int(operators.prod(self.shape)), self.shape, backend=self.backend
            )
        self.grad += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def zero_grad_(self) -> None:  # pragma: no cover
        """
        Reset the derivative on this variable.
        """
        self.grad = None
