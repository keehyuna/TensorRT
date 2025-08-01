"""
.. _auto_generate_converters:

Automatically Generate a Converter for a Custom Kernel
===================================================================

We are going to demonstrate how to automatically generate a converter for a custom kernel using Torch-TensorRT using
the new Python based plugin system in TensorRT 10.8.

Torch-TensorRT supports falling back to PyTorch implementations of operations in the case that Torch-TensorRT
does not know how to compile them in TensorRT. However, this comes at the cost of a graph break and will reduce the performance of the model.
The easiest way to fix lack of support for ops is by adding a decomposition (see:
`Writing lowering passes for the Dynamo frontend <https://pytorch.org/TensorRT/contributors/writing_dynamo_aten_lowering_passes.html>`_) - which defines the operator
in terms of PyTorch ops that are supported in Torch-TensorRT or a converter (see:
`Writing converters for the Dynamo frontend <https://pytorch.org/TensorRT/contributors/dynamo_converters.html>`_) - which defines the operator in terms of TensorRT operators.

In some cases there isn't a great way to do either of these, perhaps because the operator is a custom kernel that is not part of standard PyTorch or
TensorRT cannot support it natively.

For these cases, it is possible to use a TensorRT plugin to replace the operator **inside** the TensorRT engine, thereby avoiding
the performance and resource overhead from a graph break.

Previously this involved a complex process in not only building a performant kernel but setting it up to run in TensorRT (see: `Using Custom Kernels within TensorRT Engines with Torch-TensorRT <https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/custom_kernel_plugins.html>`_).
With TensorRT 10.8, there is a new Python native plugin system which greatly streamlines this process. This
plugin system also allows Torch-TensorRT to automatically generate the necessary conversion code to convert the
operation in PyTorch to TensorRT.
"""

# %%
# Writing Custom Operators in PyTorch
# -----------------------------------------
#
#  Pervious tutorials already cover creating custom operators in PyTorch which later get used with Torch-TensorRT.
# Here we define a simple elementwise multiplication operator in Triton. This operator is then registered as a custom op in PyTorch.
# with its host launch code as well as a "meta-kernel", A meta-kernel is a function that describes the shape and data type
# transformations that the operator will perform. This meta-kernel is used by Dynamo and Torch-TensorRT, so it
# is necessary to define.
#

from typing import Tuple

import tensorrt.plugin as trtp
import torch
import torch_tensorrt
import triton
import triton.language as tl


@triton.jit
def elementwise_mul_kernel(X, Y, Z, BLOCK_SIZE: tl.constexpr):
    # Program ID determines the block of data each thread will process
    pid = tl.program_id(0)
    # Compute the range of elements that this thread block will work on
    block_start = pid * BLOCK_SIZE
    # Range of indices this thread will handle
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load elements from the X and Y tensors
    x_vals = tl.load(X + offsets)
    y_vals = tl.load(Y + offsets)
    # Perform the element-wise multiplication
    z_vals = x_vals * y_vals
    # Store the result in Z
    tl.store(Z + offsets, z_vals)


@torch.library.custom_op("torchtrt_ex::elementwise_mul", mutates_args=())  # type: ignore[misc]
def elementwise_mul(
    X: torch.Tensor, Y: torch.Tensor, b: float = 0.2, a: int = 2
) -> torch.Tensor:
    # Ensure the tensors are on the GPU
    assert X.is_cuda and Y.is_cuda, "Tensors must be on CUDA device."
    assert X.shape == Y.shape, "Tensors must have the same shape."

    # Create output tensor
    Z = torch.empty_like(X)

    # Define block size
    BLOCK_SIZE = 1024

    # Grid of programs
    grid = lambda meta: (X.numel() // meta["BLOCK_SIZE"],)

    # Launch the kernel
    elementwise_mul_kernel[grid](X, Y, Z, BLOCK_SIZE=BLOCK_SIZE)

    return Z


# %%
# The meta kernel for an elementwise operation is just the shape and dtype of one of the inputs since we will not change the shape
# in the course of the operation.


@torch.library.register_fake("torchtrt_ex::elementwise_mul")
def _(x: torch.Tensor, y: torch.Tensor, b: float = 0.2, a: int = 2) -> torch.Tensor:
    return x


# %%
# Writing Plugins for TensorRT using the Quick Deploy Plugin system
# -------------------------------------------------------------------
# The quick deployment plugin system in TensorRT 10.8 allows for the creation of custom plugins in Python with significantly
# less boilerplate. It uses a similar system PyTorch where you define a function that describes the shape and data type transformations
# that the operator will perform and then define the code to launch the kernel given GPU memory handles.
#


# %%
# Just like the PyTorch meta kernel, there is no transformation in shape or data type between the input and output so
# we can just tell TensorRT to expect the same shape as we get in
#
@trtp.register("torchtrt_ex::elementwise_mul")
def _(
    x: trtp.TensorDesc, y: trtp.TensorDesc, b: float, a: int
) -> Tuple[trtp.TensorDesc]:
    return x.like()


# %%
# Here we reuse similar host launch code as PyTorch but we need to convert the TensorRT tensors into PyTorch tensors prior to launching the kernel
# These operations are also in-place, so the result must be put in the the output tensors provided by TensorRT.
@trtp.impl("torchtrt_ex::elementwise_mul")
def _(
    x: trtp.Tensor,
    y: trtp.Tensor,
    b: float,
    a: int,
    outputs: Tuple[trtp.Tensor],
    stream: int,
):
    # Define block size
    BLOCK_SIZE = 1024

    # Grid of programs
    grid = lambda meta: (x.numel() // meta["BLOCK_SIZE"],)

    x_t = torch.as_tensor(x, device="cuda")
    y_t = torch.as_tensor(y, device="cuda")
    z_t = torch.as_tensor(outputs[0], device="cuda")
    # Launch the kernel
    elementwise_mul_kernel[grid](x_t, y_t, z_t, BLOCK_SIZE=BLOCK_SIZE)


# %%
# Generating the Converter
# -------------------------------------------------------------------
# Given that we have defined the custom operator in PyTorch and TensorRT, we can now generate the converter for the operation.
# As long as the namespace and names match, the following function will automatically generate the converter for the operation.
torch_tensorrt.dynamo.conversion.plugins.generate_plugin_converter(
    "torchtrt_ex::elementwise_mul", supports_dynamic_shapes=True
)


# %%
# Using our converter with a model
# -------------------------------------------------------------------
#
# Now we can use our custom operator in a model and compile it with Torch-TensorRT.
# We can see that the custom operator is used as one of the operations in the forward pass of the model.
# The process of compiling the model at this point is identical to standard Torch-TensorRT usage.
class MyModel(torch.nn.Module):  # type: ignore[misc]
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.add(x, y)
        res = torch.ops.torchtrt_ex.elementwise_mul.default(x, z, a=1)

        return res


my_model = MyModel().to("cuda")
m = torch.full((64, 64), 2, device="cuda", dtype=torch.float)
n = torch.full((64, 64), 3, device="cuda", dtype=torch.float)

with torch_tensorrt.logging.errors():
    model_trt = torch_tensorrt.compile(my_model, inputs=[m, n], min_block_size=1)
    for i in range(300):
        res = model_trt(m, n)
        assert torch.allclose(res, my_model(m, n))

print("Ran with custom plugin!")
