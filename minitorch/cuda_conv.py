from typing import Tuple

import numpy as np
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function
from .autodiff import (
    Context
)
from .cuda_ops import tensor_matrix_multiply

def im2col(
    input: Tensor,
    kernel: Tensor,
    stride: int,
    padding: int
):
    b, c, h, w = input.shape
    out_c, in_c, kh, kw = kernel.shape
    tensor_data = input._tensor
    storage = tensor_data._storage
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    
    storage = storage.reshape(tensor_data.shape)
    storage = np.pad(storage, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    
    stride_b = h_out * w_out * c * kh * kw
    stride_ck = c * kh * kw
    out = np.zeros((b * stride_b))

    for curr_batch in range(b):
        idx = 0
        for curr_h in range(h_out):
            for curr_w in range(w_out):
                patch = storage[
                    curr_batch, 
                    :, 
                    curr_h*stride : curr_h*stride+kh,
                    curr_w*stride : curr_w*stride+kw
                    ]
                
                out_pos = (
                    curr_batch * stride_b + 
                    curr_h * (w_out * stride_ck) +
                    curr_w * stride_ck
                )
                out[out_pos : out_pos + (c * kh * kw)] = patch.reshape(-1)
                idx += 1
    
    out = TensorData(out, (b, h_out * w_out, c * kh * kw), (stride_b, stride_ck, 1))
    return input._new(out), h_out, w_out


def _cuda_tensor_conv2d(
    input: Tensor, 
    kernel: Tensor, 
    stride: int, 
    padding: int,
):
    batch, channels, h, w = input.shape
    out_channels, in_channels, kh, kw = kernel.shape
    assert channels == in_channels
    input, h_out, w_out = im2col(input, kernel, stride, padding)
    # Reshape kernel to [c*kh*kw, out_channels]
    kernel = kernel.contiguous().view(out_channels, in_channels * kh * kw).permute(1, 0).contiguous()
    output = input @ kernel
    output = output.permute(0, 2, 1).contiguous()
    return output.view(batch, out_channels, h_out, w_out)

class CudaConv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor, stride=1, padding=1) -> Tensor:
        output = _cuda_tensor_conv2d(input, kernel, stride, padding)
        ctx.save_for_backward(input, kernel, stride, padding, output.shape)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, kernel, stride, padding, output_shape = ctx.saved_tensors
        
        # Compute grad of input
        flipped_kernel = kernel.flip(dims=(2, 3))
        grad_input = _cuda_tensor_conv2d(grad_output, flipped_kernel, stride, padding)
        
        # Compute grad of kernel
        grad_kernel = _cuda_tensor_conv2d(input, grad_output, stride=1, padding=0)
        
        return grad_input, grad_kernel
    
conv2d_cuda = CudaConv2dFun.apply