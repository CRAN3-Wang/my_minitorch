from typing import Tuple

import numpy as np
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    TensorData,
)
from .tensor_functions import Function
from .autodiff import (
    Context
)
from .cuda_ops import tensor_matrix_multiply

@cuda.jit
def im2col_cuda_kernel(
    padded_input: Storage, 
    out: Storage, 
    b: int, c: int,
    kh: int, kw: int,
    h_out: int, w_out: int,
    stride: int
    ):
    """
    The cuda kernele of im2col func

    Args:
        padded_input (Storage): padded_input
        out (Storage): out storage
        b (int): batch size
        c (int): channels
        kh (int): kernel height
        kw (int): kernel width
        h_out (int): output height
        w_out (int): output width
        stride (Strides): stride of conv2d
    """    
    
    # Thread and block indexing
    batch_idx = cuda.blockIdx.x
    
    # Calculate global thread index
    thread_idx = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    
    # Check if thread is within output dimensions
    if batch_idx < b and thread_idx < (h_out * w_out):
        # Calculate output height and width indices
        curr_h = thread_idx // w_out
        curr_w = thread_idx % w_out
        
        # Stride calculations
        stride_ck = c * kh * kw
        stride_b = h_out * w_out * stride_ck
        
        # Calculate starting output position of this patch
        out_pos = (
            batch_idx * stride_b + 
            thread_idx * stride_ck
        )
        
        # Extract patch
        for ch in range(c):
            for kh_idx in range(kh):
                for kw_idx in range(kw):
                    # Calculate input indices with stride and padding
                    input_h = curr_h * stride + kh_idx
                    input_w = curr_w * stride + kw_idx
                    
                    # Store flattened patch
                    out[out_pos] = padded_input[batch_idx, ch, input_h, input_w]
                    out_pos += 1

def im2col_cuda(
    input: Tensor,
    kernel: Tensor,
    stride: int,
    padding: int
)-> Tensor:
    """
    cuda version of im2col, it is a preparation for conv2d

    Args:
        input (Tensor): input tensor
        kernel (Tensor): kernel tensor
        stride (int): stride of conv2d
        padding (int): padding of conv2d

    Returns:
        Tensor: input col tensor on cuda
    """    
    # Some shapes
    b, c, h, w = input.shape
    out_c, in_c, kh, kw = kernel.shape
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    
    # Find storage of input tensor and padding
    tensor_data = input._tensor
    storage = tensor_data._storage
    padded_input = storage.reshape(tensor_data.shape)
    padded_input = np.pad(padded_input, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    padded_input = cuda.to_device(padded_input)
    
    # Malloc for out tensor
    total_elements = b * h_out * w_out * c * kh * kw
    out = cuda.device_array((total_elements,), dtype=np.float64)
    
    # The following is the launching process of cuda kernel
    # A grid. The each row is a batch, therefor every colunme represents a part of the matrix
    BLOCK_DIM = 32
    GRID_DIMX = b
    GRID_DIMY = -(-(h_out * w_out) // BLOCK_DIM)
    
    im2col_cuda_kernel[
        (GRID_DIMX, GRID_DIMY),
        BLOCK_DIM
    ](
        padded_input,
        out,
        b,
        c,
        kh,
        kw,
        h_out,
        w_out,
        stride
    )
    
    out = TensorData(
        out, 
        (b, h_out * w_out, c * kh * kw), 
        (h_out * w_out * c * kh * kw, c * kh * kw, 1)
        )
    return input._new(out), h_out, w_out


def _cuda_tensor_conv2d(
    input: Tensor, 
    kernel: Tensor, 
    stride: int, 
    padding: int,
) -> Tensor:
    """
    The cuda version of tensor conv2d

    Args:
        input (Tensor): input tensor
        kernel (Tensor): kernel of conv2d
        stride (int): stide
        padding (int): padding

    Returns:
        Tensor: result of conv2d
    """    
    # Some shapes
    batch, channels, h, w = input.shape
    out_channels, in_channels, kh, kw = kernel.shape
    assert channels == in_channels
    
    # The following performs conv2d
    input, h_out, w_out = im2col_cuda(input, kernel, stride, padding)
    # Reshape kernel to [c*kh*kw, out_channels]
    kernel = kernel.contiguous().view(out_channels, in_channels * kh * kw).permute(1, 0).contiguous()
    output = input @ kernel
    output = output.permute(0, 2, 1).contiguous()
    return output.view(batch, out_channels, h_out, w_out), input


def _cuda_kernel_grad(
    input_col: Tensor, 
    grad_output: Tensor, 
    kernel_shape: Tuple[int, int, int, int]
) -> Tensor:
    """
    The grad w.r.t. kernel of conv2d

    Args:
        input_col (Tensor): input after im2col
        grad_output (Tensor): grad w.r.t. output of conv2d
        kernel_shape (Tuple[int, int, int, int]): kernel shape

    Returns:
        Tensor: grad
    """    
    out_c, in_c, kh, kw = kernel_shape
    # Shape inputcol [b, h_out*w_out, c*kh*kw]
    
    # Shape grad_output_col [b, out_channels, h_out*w_out]
    grad_output_shape = grad_output.shape
    grad_output_col = grad_output.view(grad_output_shape[0], grad_output_shape[1], grad_output_shape[2] * grad_output_shape[3])
    
    # Return [b, out_channels, c*kh*kw]
    grad_kernel = grad_output_col @ input_col
    # Reduce sum along batch axis
    grad_kernel = grad_kernel.sum(dim=0)
    
    if not grad_kernel._tensor.is_contiguous():
        grad_kernel = grad_kernel.contiguous()
    
    grad_kernel = grad_kernel.view(out_c, in_c, kh, kw)
    return grad_kernel


class CudaConv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, kernel: Tensor, stride=1, padding=1) -> Tensor:
        output, input_col = _cuda_tensor_conv2d(input, kernel, stride, padding)
        ctx.save_for_backward(input_col, kernel, stride, padding)
        return output
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input_col, kernel, stride, padding = ctx.saved_tensors
        
        # Compute grad of input
        flipped_kernel = kernel.flip(dims=(2, 3))
        flipped_kernel = flipped_kernel.permute(1, 0, 2, 3).contiguous()
        grad_input, _ = _cuda_tensor_conv2d(grad_output, flipped_kernel, stride, padding)

        # Compute grad of kernel
        grad_kernel = _cuda_kernel_grad(input_col, grad_output, tuple(kernel.shape))

        return grad_input, grad_kernel
    
conv2d_cuda = CudaConv2dFun.apply