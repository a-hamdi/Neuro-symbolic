import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute C = A @ B using Triton's GEMM kernel.
    
    This kernel is optimized for matrix multiplication operations.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    
    # Number of program ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Number of programs working on different parts of the output
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    # Group ID
    group_id = pid // num_pid_in_group
    
    # First program ID in this group
    first_pid_m = group_id * GROUP_SIZE_M
    
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group has fewer
    # programs working on M dimension
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    # Program ID in the group
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Block start indices for this program
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create block pointers for A and B
    a_block_ptr = a_ptr + m_start * stride_am
    b_block_ptr = b_ptr + n_start * stride_bn
    
    # Initialize the accumulator to zero
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute a block of the C matrix
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Compute the block for this k iteration
        k_start = k * BLOCK_SIZE_K
        
        # Check if we need a smaller block for the last k iteration
        k_end = min(K, k_start + BLOCK_SIZE_K)
        
        # Load blocks from A and B
        a_block = tl.load(
            a_block_ptr + k_start * stride_ak,
            mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < M - m_start,
            other=0.0
        )
        
        b_block = tl.load(
            b_block_ptr + k_start * stride_bk,
            mask=tl.arange(0, BLOCK_SIZE_K)[None, :] < k_end - k_start,
            other=0.0
        )
        
        # Compute the matrix multiplication for this block
        accumulator += tl.dot(a_block, b_block)
        
    # Write the result to the output matrix C
    c_block_ptr = c_ptr + m_start * stride_cm + n_start * stride_cn
    tl.store(
        c_block_ptr,
        accumulator,
        mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < M - m_start
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using the Triton GEMM kernel.
    
    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        
    Returns:
        Output tensor of shape (M, N)
    """
    # Check input shapes
    assert a.dim() == 2, "A must be a 2D tensor"
    assert b.dim() == 2, "B must be a 2D tensor"
    assert a.shape[1] == b.shape[0], f"Incompatible dimensions: {a.shape} and {b.shape}"
    
    # Get dimensions
    M, K = a.shape
    K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Set grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    GROUP_SIZE_M = 8
    
    # Compute number of blocks needed
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # Launch the Triton kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c


@triton.jit
def logical_and_kernel(
    # Pointers to input and output tensors
    a_ptr, b_ptr, c_ptr,
    # Size of the tensor
    n_elements,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute C = A & B element-wise using Triton.
    
    This kernel is optimized for logical AND operations on binary tensors.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start index
    block_start = pid * BLOCK_SIZE
    
    # Create offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load inputs
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0)
    
    # Compute logical AND (both must be > 0)
    c = tl.where((a > 0) & (b > 0), 1.0, 0.0)
    
    # Store result
    tl.store(c_ptr + offsets, c, mask=mask)


def triton_logical_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A & B element-wise using the Triton kernel.
    
    Args:
        a: Input tensor
        b: Input tensor of same shape as a
        
    Returns:
        Binary tensor where each element is 1 if both a and b are non-zero at that position, 0 otherwise
    """
    # Check input shapes
    assert a.shape == b.shape, f"Input tensors must have the same shape: {a.shape} vs {b.shape}"
    
    # Convert inputs to same device and dtype if needed
    if a.device != b.device:
        b = b.to(a.device)
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    
    # Flatten tensors
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    # Allocate output tensor
    c_flat = torch.empty_like(a_flat)
    
    # Get number of elements
    n_elements = a_flat.numel()
    
    # Set block size
    BLOCK_SIZE = 1024
    
    # Compute number of blocks needed
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    logical_and_kernel[grid](
        a_flat, b_flat, c_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape output to original shape
    return c_flat.reshape(a.shape)


@triton.jit
def cosine_similarity_kernel(
    # Pointers to input and output tensors
    a_ptr, b_ptr, c_ptr,
    # Size of the tensor
    batch_size, vector_size,
    # Strides
    stride_ab, stride_av,
    stride_bb, stride_bv,
    stride_c,
    # Meta-parameters
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_V: tl.constexpr,
):
    """
    Compute cosine similarity between vectors in A and B.
    
    This kernel is optimized for computing similarity between embedding vectors.
    """
    # Program ID
    pid = tl.program_id(axis=0)
    
    # Block start indices
    batch_idx = pid * BLOCK_SIZE_B
    
    # Create ranges
    batch_range = batch_idx + tl.arange(0, BLOCK_SIZE_B)
    vec_range = tl.arange(0, BLOCK_SIZE_V)
    
    # Create mask for bounds checking
    batch_mask = batch_range < batch_size
    
    # Initialize accumulators
    dot_products = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)
    a_norms = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)
    b_norms = tl.zeros([BLOCK_SIZE_B], dtype=tl.float32)
    
    # Compute dot products and norms
    for v_start in range(0, vector_size, BLOCK_SIZE_V):
        # Create mask for vector dimension
        vec_mask = v_start + vec_range < vector_size
        
        # Combined mask
        mask = batch_mask[:, None] & vec_mask[None, :]
        
        # Load blocks from A and B
        a_block = tl.load(
            a_ptr + batch_range[:, None] * stride_ab + (v_start + vec_range[None, :]) * stride_av,
            mask=mask, other=0.0
        )
        
        b_block = tl.load(
            b_ptr + batch_range[:, None] * stride_bb + (v_start + vec_range[None, :]) * stride_bv,
            mask=mask, other=0.0
        )
        
        # Update dot products and norms
        dot_products += tl.sum(a_block * b_block, axis=1)
        a_norms += tl.sum(a_block * a_block, axis=1)
        b_norms += tl.sum(b_block * b_block, axis=1)
    
    # Compute cosine similarity
    similarity = dot_products / (tl.sqrt(a_norms) * tl.sqrt(b_norms) + 1e-8)
    
    # Store result
    tl.store(c_ptr + batch_range, similarity, mask=batch_mask)


def triton_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between vectors in A and B using the Triton kernel.
    
    Args:
        a: Input tensor of shape (batch_size, vector_size)
        b: Input tensor of shape (batch_size, vector_size)
        
    Returns:
        Tensor of shape (batch_size,) containing cosine similarities
    """
    # Check input shapes
    assert a.dim() == 2, "A must be a 2D tensor"
    assert b.dim() == 2, "B must be a 2D tensor"
    assert a.shape == b.shape, f"Input tensors must have the same shape: {a.shape} vs {b.shape}"
    
    # Get dimensions
    batch_size, vector_size = a.shape
    
    # Allocate output tensor
    c = torch.empty(batch_size, device=a.device, dtype=torch.float32)
    
    # Set block sizes
    BLOCK_SIZE_B = 32
    BLOCK_SIZE_V = 128
    
    # Compute number of blocks needed
    grid = (triton.cdiv(batch_size, BLOCK_SIZE_B),)
    
    # Launch the Triton kernel
    cosine_similarity_kernel[grid](
        a, b, c,
        batch_size, vector_size,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0),
        BLOCK_SIZE_B=BLOCK_SIZE_B,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )
    
    return c


# Example usage and benchmark
if __name__ == "__main__":
    import time
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set problem size
    M, K, N = 1024, 1024, 1024
    
    # Create random matrices
    a = torch.randn((M, K), device=device, dtype=torch.float32)
    b = torch.randn((K, N), device=device, dtype=torch.float32)
    
    # Warm up
    torch_result = a @ b
    triton_result = triton_matmul(a, b)
    
    # Verify correctness
    torch.testing.assert_close(torch_result, triton_result, rtol=1e-2, atol=1e-2)
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 10
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = triton_matmul(a, b)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10
    
    # Print results
    print(f"Matrix multiplication ({M}x{K} @ {K}x{N})")
    print(f"PyTorch time: {torch_time:.4f}s")
    print(f"Triton time: {triton_time:.4f}s")
    print(f"Speedup: {torch_time / triton_time:.2f}x")
    
    # Test logical AND
    a_bin = torch.randint(0, 2, (1024, 1024), device=device, dtype=torch.float32)
    b_bin = torch.randint(0, 2, (1024, 1024), device=device, dtype=torch.float32)
    
    # PyTorch implementation
    torch_and = (a_bin > 0) & (b_bin > 0)
    triton_and = triton_logical_and(a_bin, b_bin) > 0.5
    
    # Verify correctness
    torch.testing.assert_close(torch_and.float(), triton_and.float())
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = (a_bin > 0) & (b_bin > 0)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 10
    
    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = triton_logical_and(a_bin, b_bin)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 10
    
    # Print results
    print(f"\nLogical AND (1024x1024)")
    print(f"PyTorch time: {torch_time:.4f}s")
    print(f"Triton time: {triton_time:.4f}s")
    print(f"Speedup: {torch_time / triton_time:.2f}x") 