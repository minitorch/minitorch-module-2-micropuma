from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    args = list(vals)
    arg_left = args.copy()
    arg_right = args.copy()

    arg_left[arg] = arg_left[arg] + epsilon
    arg_right[arg] = arg_right[arg] - epsilon

    return (f(*arg_left) - f(*arg_right)) / (2 * epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # 存储最后的拓扑序，按照从ouput到input leaf的顺序做拓扑
    result = []
    visited = []

    # 内嵌深度搜索递归函数
    def visit(cur : Variable):
        # constant 不参与反向传播，没有history信息
        if cur.is_constant():
            return
        # 判断是否访问过
        elif cur.unique_id in visited:
            return
        else:
            visited.append(cur.unique_id)
            for next in cur.history.inputs:
                visit(next)
            result.insert(0, cur)
    visit(variable)
    return result;
    


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    top_order = topological_sort(variable)

    # 初始输出节点的node to grad
    # i -> j; i -> k
    # 则node_to_grad[i] = Vi-j + Vi-k
    node_to_grad = {}
    node_to_grad[variable.unique_id] = deriv

    for var in top_order:
        if var.is_leaf():
            # leaf 在input遍历的时候会更新
            continue

        if var.unique_id in node_to_grad.keys():
            dv_list = node_to_grad[var.unique_id]
            print("dv_list is: ", dv_list)
        else:
            dv_list = deriv
        inputs = var.chain_rule(dv_list)

        for key, dv in inputs:
            # 如果是叶子节点，则直接在叶子上做累积即可
            if key.is_leaf(): 
                key.accumulate_derivative(dv)            
                print("result is:", key.derivative)
                continue
            
            if key.unique_id in node_to_grad.keys():
                node_to_grad[key.unique_id] += dv
                tmp = node_to_grad[key.unique_id]
                print(f"unique id is : {key.unique_id}")
                print("value is : ", tmp)
            else:
                node_to_grad[key.unique_id] = dv
                print(f"unique id is : {key.unique_id}")

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    # 在当前的context中存反向传播需要的values
    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
