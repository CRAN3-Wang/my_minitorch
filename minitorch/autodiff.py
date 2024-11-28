from dataclasses import dataclass
from typing import Any, Iterable, Tuple

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
    val_l = vals[arg] - epsilon
    val_r = vals[arg] + epsilon
    
    tup_l = vals[:arg] + (val_l,) + vals[arg + 1:]
    tup_r = vals[:arg] + (val_r,) + vals[arg + 1:]
    
    f_l = f(tup_l)
    f_r = f(tup_r)
    
    return (f_r - f_l) / (2 * epsilon)


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
    Computes the topological order of the computation graph using a depth-first search (DFS) approach.
    This function ensures that each variable is processed only after all variables that depend on it have been processed.

    Args:
        variable: The right-most variable from which to start the sort, typically the final output variable of the graph.

    Returns:
        An iterable of non-constant Variables in topological order, starting from the given variable and moving backwards.
    """
    visited = set()  # Set to keep track of visited nodes
    stack = []  # Stack to hold the topologically sorted variables

    def dfs(v: Variable) -> None:
        """Helper function to perform DFS"""
        if v.unique_id in visited:
            return
        visited.add(v.unique_id)

        # Iterate over the parents of the current variable
        for parent in v.parents:
            if not parent.is_constant():  # Only consider non-constant variables
                dfs(parent)

        stack.append(v)  # Append the variable to the stack after processing its parents

    # Start DFS from the given variable
    dfs(variable)

    # Since we want the elements in topological order starting from the right,
    # we need to reverse the stack because the deepest dependent variables are at the top.
    return reversed(stack)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives_dict = {variable.unique_id: deriv}
    
    sorted_list = topological_sort(variable)
    
    for curr_variable in sorted_list:
        if curr_variable.is_leaf():
            continue
        curr_derivatives = curr_variable.chain_rule(derivatives_dict[curr_variable.unique_id])
        
        for this_variable, this_derivative in curr_derivatives:
            if this_variable.is_leaf():
                this_variable.accumulate_derivative(this_derivative)  
            else:
                if this_variable.unique_id not in derivatives_dict:
                    derivatives_dict[this_variable.unique_id] = this_derivative  
                else:
                    derivatives_dict[this_variable.unique_id] += this_derivative


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
