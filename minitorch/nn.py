from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .cuda_ops import CudaOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor
import numpy as np


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    new_height = height / kh
    new_width = width / kw
    
    out = input.contiguous().view(batch, channel, height, new_width, kw)
    out = out.permute(0, 1, 3, 2, 4)
    out = out.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    out, new_height, new_width = tile(input, kernel)
    out = out.mean(dim=4)
    return out.view(batch, channel, new_height, new_width)


# max_reduce = FastOps.reduce(operators.max, -1e9)
max_reduce = CudaOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_tensors
        return grad_output * argmax(input, int(dim.item())), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    flatten_shape = int(np.prod(input.shape))
    shifted_input = input - max(
        input.contiguous().view(
            flatten_shape,
        ),
        0,
    )

    # compute the exponentiels values
    exp_input = shifted_input.exp()

    # compute the sum of exponentiels along the specified dimension
    sum_exp_input = exp_input.sum(dim)

    # normalize by dividing by the exponentiels by the sum of exponentiels
    softmax_input = exp_input / sum_exp_input

    return softmax_input


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    max_tensor = max(input, dim)

    # substract the max from the input
    shifted_input = input - max_tensor

    # compute the exponentiels values
    exp_input = shifted_input.exp()

    # sum the exponential.
    sum_input = exp_input.sum(dim)

    # take the log.
    log_input = sum_input.log()

    # add the max
    log_sum_exp_input = log_input + max_tensor

    return input - log_sum_exp_input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    out, new_height, new_width = tile(input, kernel)
    out = max(out, dim=4)
    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
