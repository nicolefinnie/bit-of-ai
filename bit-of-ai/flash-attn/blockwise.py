"""Conceptual blockwise attention in PyTorch as a tutorial.
"""
import torch
import time 

def blockwise_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        B_r: int=16,
        B_c:int=16
    ) -> torch.Tensor:
    """Blockwise attention for scaled dot-product attention.
    
    Computes a block-wise attention mechanism over input query (Q), key (K), and value (V) matrices.
    This implementation handles large matrices by splitting them into smaller blocks to optimize memory usage.

    Args:
        Q (torch.Tensor): Query matrix of shape (N_out, d), where N_out is the output dimension and d is the feature dimension.
        K (torch.Tensor): Key matrix of shape (N_inp, d), where N_inp is the input dimension and d is the feature dimension.
        V (torch.Tensor): Value matrix of shape (N_inp, d), where N_inp is the input dimension and d is the feature dimension.
        B_r (int, optional): Block size for rows of Q and O (default is 16).
        B_c (int, optional): Block size for columns of K and V (default is 16).

    Returns:
        torch.Tensor: Output matrix of shape (N_out, d), resulting from the block attention.

    Process:
        - The input matrices Q, K, and V are split into smaller blocks of size `B_r` (for rows) and `B_c` (for columns).
        - It computes scaled dot-product attention in a block-wise manner:
            1. Compute scaled attention scores S_ij = (Q_i @ K_j^T) for each pair of blocks (i, j).
            2. Apply scaling by a factor of 1/sqrt(d) to the attention scores.
            3. Perform softmax-like scaling and normalization across the blocks.
            4. Accumulate the weighted sum of values (V_j) based on the attention scores.
            5. Handle memory constraints by iteratively accumulating intermediate results for O_i (output)

    Notes:
        - The algorithm uses exponential scaling to handle potential numerical instabilities during softmax calculations.
        - Attention scores are accumulated incrementally over blocks, and scaling factors are adjusted between iterations.
        - Tiling reduces memory usage by working with smaller submatrices at a time.

    Example:
        N_inp = 64
        N_out = 64
        d = 128

        Q = torch.randn(N_out, d)
        K = torch.randn(N_inp, d)
        V = torch.randn(N_inp, d)

        O = blockwise_attention(Q, K, V, B_r=16, B_c=16)
    """
    # number of output tokens and features, that correspond to the query vectors
    N_out, d = Q.shape
    # number of input tokens of key-value pairs and features
    N_inp, _ = K.shape
    O = torch.zeros(N_out, d, device=Q.device)
    # normalizing the attention score to each query (Vaswani et al. 2017) to ensure Q@K doesn't explode
    scaling_factor = 1/torch.sqrt(torch.tensor(d, dtype=Q.dtype, device=Q.device))  # Scaling factor for attention scores

    # number of row blocks, column blocks
    T_r = (N_out + B_r - 1 ) // B_r
    T_c = (N_inp + B_c - 1) // B_c
  
    # Iterate over every query (row major)
    for i in range(T_r):
        Q_i = Q[i*B_r:(i+1)*B_r]
        O_i = torch.zeros(B_r, d, device=Q.device) # output block
        m_i = torch.full((B_r, 1), -torch.inf, device=Q.device) # maximum for softmax stabilization

        # Iterate over every key (column major)
        for j in range(T_c):
            K_j = K[j*B_c:(j+1)*B_c]
            V_j = V[j*B_c:(j+1)*B_c]

            # query-key dot product, [B_r, B_c]
            # Compute attention scores for this block
            S_ij = scaling_factor * (Q_i @ K_j.T)
            # Prevent numerical stability
            m_i = torch.maximum(m_i, S_ij.max(dim=-1, keepdim=True).values)

            # Subtract maximum for numerical stability
            P_ij = torch.exp(S_ij - m_i)

            # Normalize attention scores
            P_ij = P_ij / P_ij.sum(dim=-1, keepdim=True)

            # Compute weighted sum of values
            O_i += P_ij @ V_j
  
        O[i*B_r:(i+1)*B_r] = O_i

    return O

def regular_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:

    scaling_factor = 1 / torch.sqrt(torch.tensor(Q.shape[-1],dtype=Q.dtype, device=Q.device))
    
    S = (Q @ K.T) * scaling_factor
    P = torch.softmax(S, dim=-1)
    O = P @ V

    return O


def test():
    torch.manual_seed(42)
    N_inp = 128
    N_out = 128
    d = 128
    B_c = 16
    B_r = 16

    Q = torch.randn(N_out, d)
    K = torch.randn(N_inp, d)
    V = torch.randn(N_inp, d)
   
    start = time.time()
    blockwise_attention(Q, K, V, B_r=B_r, B_c=B_c)
    # It appears slower in python because the blocks are supposed to compute in parallel in kernel
    # But in python, it is computed sequentially, I only wanted to show the concept
    # Attention computation is not a compute-bound problem but memory-bound, that's why flash attention
    # tiles the computation so the compute unit would utilize cache if the blocks can fit in cache
    # cache has way bigger throughput than memory
    print(f"Blockwise attention score time: {time.time() - start}")
   
    start = time.time()
    regular_attention(Q, K, V)
    print(f"Regular attention score: {time.time() - start}")
   
if __name__=='__main__':
    test()
