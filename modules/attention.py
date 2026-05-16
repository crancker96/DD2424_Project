import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # query shape: [bs, heads, seq_len, 64]
    d_k = query.size(3)
    seq_len = query.size(2)

    # computing scaled dot product to get similarity/relevance scores
    e_scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # create and apply casual mask, 1s above diagonal, blocks future positions
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1).bool()
    e_scores = e_scores.masked_fill(causal_mask, -10000)

    # adds padding mask 
    e_scores = e_scores + attention_mask

    # softmax to get sum of 1, scores set to -10000 will be 0 in softmax exp(-inf) = 0
    attention_weights = torch.softmax(e_scores, dim=-1)

    # used dropeout rate from config of 0.1
    attention_weights = self.dropout(attention_weights)

    # weighted sum of values
    attention_output = torch.matmul(attention_weights, value)

    # merges all heads back reverse to what was done in transform()
    attention_output = rearrange(attention_output, 'b h t d -> b t (h d)')
    return attention_output

  def torch_flash_attention(self, key, query, value, attention_mask, causal=True):

    # query/key/value:
    # [B, H, N, D]

    out = nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=True
    )
    B, H, N, D = out.shape

    # transpose to [B, N, H, D]
    out = out.transpose(1, 2)

    # merge heads
    out = out.contiguous().view(B, N, H * D)
    return out
  
  def flash_attention(self, key, query, value, attention_mask, causal=True):
    """
    query, key, value:
        [B, H, N, D]
    """

    B, H, N, D = query.shape

    device = query.device

    # tile sizes
    B_c = min(128, N)
    B_r = min(128, N)

    T_c = (N + B_c - 1) // B_c
    T_r = (N + B_r - 1) // B_r

    scale = D ** -0.5

    O = torch.zeros_like(query)

    # running stats
    l = torch.zeros(B, H, N, device=device)
    m = torch.full((B, H, N), -float("inf"), device=device)

    for i in range(T_r):

      r_start = i * B_r
      r_end = min((i + 1) * B_r, N)

      q = query[:, :, r_start:r_end, :]
      O_i = O[:, :, r_start:r_end, :].clone()

      l_i = l[:, :, r_start:r_end].clone()
      m_i = m[:, :, r_start:r_end].clone()

      for j in range(T_c):

        c_start = j * B_c
        c_end = min((j + 1) * B_c, N)

        k = key[:, :, c_start:c_end, :]
        v = value[:, :, c_start:c_end, :]

        # attention scores
        S = torch.matmul(q, k.transpose(-2, -1)) * scale
        # [B,H,B_r,B_c]

        # causal mask
        if causal:
          q_idx = torch.arange(r_start, r_end, device=device)
          k_idx = torch.arange(c_start, c_end, device=device)

          mask = q_idx[:, None] >= k_idx[None, :]
          S = S.masked_fill(~mask, -float("inf"))

        # row max
        m_ij = S.max(dim=-1).values

        # exponentials
        P = torch.exp(S - m_ij.unsqueeze(-1))

        # row sums
        l_ij = P.sum(dim=-1)

        # updated max
        m_new = torch.maximum(m_i, m_ij)

        # correction factors
        alpha = torch.exp(m_i - m_new)
        beta = torch.exp(m_ij - m_new)

        # updated normalization
        l_new = alpha * l_i + beta * l_ij

        # weighted values
        PV = torch.matmul(P, v)

        # output update
        O_i = (
            ((alpha * l_i) / l_new).unsqueeze(-1) * O_i
            + (beta / l_new).unsqueeze(-1) * PV
        )

        # save stats
        l_i = l_new
        m_i = m_new

      # write back
      O[:, :, r_start:r_end, :] = O_i
      l[:, :, r_start:r_end] = l_i
      m[:, :, r_start:r_end] = m_i

    B, H, N, D = O.shape

    # transpose to [B, N, H, D]
    O = O.transpose(1, 2)

    # merge heads
    O = O.contiguous().view(B, N, H * D)
    return O





  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.torch_flash_attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
