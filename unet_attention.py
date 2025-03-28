from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SpatialTransformer(nn.Module):
    """
    ## Spatial Transformer
    """

    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        """
        :param channels: is the number of channels in the feature map
        :param n_heads: is the number of attention heads
        :param n_layers: is the number of transformer layers
        :param d_cond: is the size of the conditional embedding
        """
        super().__init__()
        # Initial group normalization
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        # Initial 1x1 convolution
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels // n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )

        # Final 1x1 convolution
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the feature map of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """
        b, c, h, w = x.shape
        # print("SpatialTransformer: x shape:", x.shape)
        # print("SpatialTransformer: cond shape:", cond.shape if cond else "None")
        # For residual connection
        x_in = x
        # Normalize
        x = self.norm(x)
        # Initial 1x1 convolution
        x = self.proj_in(x)
        # Transpose and reshape from `[batch_size, channels, height, width]`
        # to `[batch_size, height * width, channels]`
        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        # Apply the transformer layers
        for i, block in enumerate(self.transformer_blocks):
            # print(f"SpatialTransformer: before transformer block {i}, x shape:", x.shape)
            x = block(x, cond)
            # print(f"SpatialTransformer: after transformer block {i}, x shape:", x.shape)
        # Reshape and transpose from `[batch_size, height * width, channels]`
        # to `[batch_size, channels, height, width]`
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        # Final 1x1 convolution
        x = self.proj_out(x)
        # Add residual
        return x + x_in


class BasicTransformerBlock(nn.Module):
    """
    ### Transformer Layer
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        :param d_cond: is the size of the conditional embeddings
        """
        super().__init__()
        # Self-attention layer and pre-norm layer
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        # Cross attention layer and pre-norm layer
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network and pre-norm layer
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        :param x: are the input embeddings of shape `[batch_size, height * width, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """
        # print("BasicTransformerBlock: x shape:", x.shape)
        # print("BasicTransformerBlock: cond shape:", cond.shape if cond else "None")
        # Self attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-attention with conditioning
        x = self.attn2(self.norm2(x), cond=cond) + x
        # Feed-forward network
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Module):
    """
    ### Cross Attention Layer

    This falls-back to self-attention when conditional embeddings are not specified.
    """

    use_flash_attention: bool = False

    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        """
        :param d_model: is the input embedding size
        :param n_heads: is the number of attention heads
        :param d_head: is the size of an attention head
        :param d_cond: is the size of the conditional embeddings
        :param is_inplace: specifies whether to perform the attention softmax computation inplace to
            save memory
        """
        super().__init__()
        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head

        # Attention scaling factor
        self.scale = d_head ** -0.5

        # Attention head output dimension
        d_attn = d_head * n_heads
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)

        # Final linear layer
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))
        
        # If d_model != d_cond, add a projection for the fallback
        if d_model != d_cond:
            self.cond_proj = nn.Linear(d_model, d_cond)
        else:
            self.cond_proj = nn.Identity()

        # Setup flash attention if available
        try:
            from flash_attn.flash_attention import FlashAttention
            self.flash = FlashAttention()
            self.flash.softmax_scale = self.scale
        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        :param x: are the input embeddings of shape `[batch_size, seq, d_model]`
        :param cond: is the conditional embeddings of shape `[batch_size, n_cond, d_cond]`
        """
        has_cond = cond is not None
        # print("CrossAttention: has_cond =", has_cond)
        if not has_cond:
            cond = self.cond_proj(x)
        # print("CrossAttention: cond shape before linear (k, v):", cond.shape)
        # print("CrossAttention: Expected cond last dimension (d_cond):", self.to_k.in_features)

        # Get query, key and value vectors
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)

        # print("CrossAttention: q shape after to_q:", q.shape)
        # print("CrossAttention: k shape after to_k:", k.shape)
        # print("CrossAttention: v shape after to_v:", v.shape)

        # Use flash attention if it's available and conditions are met
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q, k, v)
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Flash Attention

        :param q: query vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        :param k: key vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        :param v: value vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        """
        batch_size, seq_len, _ = q.shape
        qkv = torch.stack((q, k, v), dim=2)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.d_head)

        if self.d_head <= 32:
            pad = 32 - self.d_head
        elif self.d_head <= 64:
            pad = 64 - self.d_head
        elif self.d_head <= 128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size {self.d_head} too large for Flash Attention')

        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.n_heads, pad)), dim=-1)

        out, _ = self.flash(qkv)
        out = out[:, :, :, :self.d_head]
        out = out.reshape(batch_size, seq_len, self.n_heads * self.d_head)
        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        #### Normal Attention
        
        :param q: query vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        :param k: key vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        :param v: value vectors before splitting heads, shape `[batch_size, seq, d_attn]`
        """
        # print("Normal Attention: q shape before splitting:", q.shape)
        # print("Normal Attention: k shape before splitting:", k.shape)
        # print("Normal Attention: v shape before splitting:", v.shape)

        # Split into heads
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(*out.shape[:2], -1)
        # print("Normal Attention: out shape after linear mapping:", out.shape)
        return self.to_out(out)


class FeedForward(nn.Module):
    """
    ### Feed-Forward Network
    """

    def __init__(self, d_model: int, d_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        # print("FeedForward: input shape:", x.shape)
        out = self.net(x)
        # print("FeedForward: output shape:", out.shape)
        return out


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        x_proj = self.proj(x)
        # Split into two parts for the GeGLU computation
        x_val, gate = x_proj.chunk(2, dim=-1)
        out = x_val * F.gelu(gate)
        # print("GeGLU: input shape:", x.shape, "proj output shape:", x_proj.shape, "output shape:", out.shape)
        return out
