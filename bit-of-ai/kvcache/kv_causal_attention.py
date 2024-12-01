"""Hello it's me again, I want to write something easy to prove how KV-Cache works in a Paged Attention manner
together. Each page attention is processed by FlashAttention. However, this is definitely not the optimal
way because it's still on the graph level. To optimize it, we need to implement paging on the kernel
of the target hardware. It also depends on whether paging is idea for your target inference hardware.
Moving data to pages could be very inefficient for target hardware."""
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config


class Conv1D(nn.Module):
    """1D convolution module as used in GPT-2, equivalent to a fully-connected layer.
    Only for debugging to ensure the Conv1D implementation is equivalent to nn.Linear.
    """
    def __init__(self, nf, nx):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))  # (out_features, in_features)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
    
      # x is (batch_size, sequence_length, input_dim), e.g., (B, T, 768)
        B, T, nx = x.size()        
        # Flatten batch and sequence dimensions: (B * T, input_dim)
        x = x.view(B * T, nx)

        # Matrix multiplication and bias addition: (B * T, 2304)
        x = torch.addmm(self.bias, x, self.weight)  # Transpose weight for correct multiplication

        # Reshape back to (B, T, output_dim), e.g., (B, T, 2304)
        x = x.view(B, T, -1)
        return x

class PagedCausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config, paging_size: int = 128):
        """A memory-efficient multi-head attention mechanism.

        Reference: This is Karpathy's implementation that is compatible with GPT2
        https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
        I tried to follow the naming convention of Kaparthy's code.

        Ensure that the hidden dimension is divisible by the number of heads
        I don't want to throw an assertion here as Kaparthy did, not cool for production
        
        """
        super().__init__()
        # Define the layers (query, key, value) 
        # We could split it to three linears is also more quantization friendly because you can
        # quantize their weights separately with different scaling factors to reduce quantization error.
        # But following the original implementation is just easier to load the pretrained weight.
        # 768 -> 3 * 768
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # only for debugging to ensure the Conv1D implementation is equivalent to nn.Linear
        # self.c_attn = Conv1D(config.n_embd * 3, config.n_embd)
        # self.c_proj = Conv1D(config.n_embd, config.n_embd)
       
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # chunk size for processing the sequence
        self.paging_size = paging_size


    def forward(self, x, layer_past = None, attention_mask=None, use_cache = False, **kwargs) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
        """Split attention to pages and process each page using FlashAttention.

           1. Split attention to multiple heads
           2. Split the sequence into pages
           3. Process each page using FlashAttention

            nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
            e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        
            Args:
                x: input tensor (B, T, C)
                layer_past: past cached key and value for fast autoregressive generation
                    if use_cache is True, layer_past is expected to be a tuple of (k, v)
                use_cache: whether to use the cached key/value for fast autoregressive generation
                    during inference: Set it to True
                    during training: Set it to False

        """
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        qkv = self.c_attn(x)
        #  (B, T, 3 * 768) -> (B, T, 768) x 3
        q,k,v = qkv.split(self.n_embd, dim=2)
        # Reshape for multi-head attention to (B, n_head, T, head_embedding_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        #print(f"Before layer_past: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        if layer_past is not None:
            past_k, past_v = layer_past[0]
            # concatenate the past sequence with the current sequence
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        # one example
        # Before layer_past: q shape: torch.Size([1, 12, 1, 64]), k shape: torch.Size([1, 12, 1, 64]), v shape: torch.Size([1, 12, 1, 64])
        # After layer_past: q shape: torch.Size([1, 12, 1, 64]), k shape: torch.Size([1, 12, 529, 64]), v shape: torch.Size([1, 12, 529, 64])
        # Present: q shape: torch.Size([1, 12, 1, 64]), k shape: torch.Size([1, 12, 529, 64]), v shape: torch.Size([1, 12, 529, 64])

        present = (k, v) if use_cache else None

        print(f"After layer_past: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
           
        # if T <= self.paging_size or use_cache:
        #     y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=True)
        # else:
        #     # Handle paging for longer sequences
        #     y = torch.zeros((B, self.n_head, T, C // self.n_head), device=q.device)
        #     num_pages = (T + self.paging_size - 1) // self.paging_size  # Ensure at least one page
        #     for i in range(num_pages):
        #         start_idx = i * self.paging_size
        #         end_idx = min((i + 1) * self.paging_size, T)
                
        #         page_query = q[:, :, start_idx:end_idx]
        #         page_key = k[:, :, start_idx:end_idx]
        #         page_value = v[:, :, start_idx:end_idx]

        #         atten_y = F.scaled_dot_product_attention(page_query, page_key, page_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=True)
        #         y[:, :, start_idx:end_idx] = atten_y
              
        # Combine heads again and project to the resulting dimension (B, T, 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)  # Add residual dropout
      
       
        # attention output, the next layer_past
        return y, present
    
        
class KVGPT2Model(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        # Replace attention layer in GPT-2  
        for i, block in enumerate(self.transformer.h):
            block.attn = PagedCausalSelfAttention(config, paging_size=128)


