import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def exists(x):
    return x is not None

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.n_embed = config.n_embed
        self.n_head = config.n_head

        self.head_dim = config.n_embed // config.n_head
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        assert C == self.n_embed # make sure the input size is correct
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, n_head, T, head_dim)
        assert k.size(-1) == self.head_dim # make sure the input size is correct

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = self.c_proj(y)
        return x

class MLP(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.c_fc = nn.Linear(n_embed, 4 * n_embed)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class MLPSiLu(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = self.silu(x)
        x = self.c_proj(x)
        return x

class LayerNormMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, scale, shift):
        x = self.ln(x)
        x = x * (1 + scale) + shift
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNormMod(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNormMod(config.n_embed)
        self.mlp = MLP(config.n_embed)

        self.time_mlp = MLPSiLu(config.sinusoidal_dim, config.n_embed * 2)
        self.class_mlp = MLPSiLu(config.sinusoidal_dim, config.n_embed * 2)

    def forward(self, x, time_emb, cl_emb, mask=None):

        scale1_scale2 = self.time_mlp(time_emb)
        scale1_scale2 = rearrange(scale1_scale2, 'b c -> b 1 c')
        scale1, scale2 = scale1_scale2.chunk(2, dim=-1)

        shift1_shift2 = self.class_mlp(cl_emb)
        shift1_shift2 = rearrange(shift1_shift2, 'b c -> b 1 c')
        shift1, shift2 = shift1_shift2.chunk(2, dim=-1)

        x = x + self.attn(self.ln_1(x, scale1, shift1), mask)
        x = x + self.mlp(self.ln_2(x, scale2, shift2))
        return x

class TokenEmb(nn.Module):
    def __init__(self, n_channels, n_embed, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_embed, kernel_size, padding=padding)

    def forward(self, x, mask=None):
        x = self.conv(x)
        if exists(mask):
            x = x * mask
        return x.permute(0, 2, 1) # (B, T, C)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class SinusoidalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoding = nn.Sequential(
            RandomOrLearnedSinusoidalPosEmb(config.learned_sinusoidal_dim, config.random_fourier_features),
            nn.Linear(config.learned_sinusoidal_dim + 1, config.sinusoidal_dim),
            nn.GELU(),
            nn.Linear(config.sinusoidal_dim, config.sinusoidal_dim)
        )

    def forward(self, x):
        return self.encoding(x)
    

@dataclass
class GPTConfig:
    max_len:int = 48
    n_timesteps:int = 1000
    n_channels:int = 21
    n_classes:int = 20
    n_layer:int = 8
    n_head:int = 8
    n_embed:int = 256
    learned_sinusoidal_dim:int = 16
    random_fourier_features:bool = True
    sinusoidal_dim:int = 256 # sinusoidal embedding dim, in Unet it is 4 * n_embed

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = config.n_channels
        self.classes = config.n_classes
        self.self_condition = False

        self.transformer = nn.ModuleDict(dict(
            tok_embedding = TokenEmb(config.n_channels, config.n_embed, 7, padding = 3), # token embedding
            pos_embedding = nn.Embedding(config.max_len, config.n_embed), # positional embedding
            time_embedding = SinusoidalEmbedding(config), # time embedding
            class_embedding = SinusoidalEmbedding(config), # class embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.n_channels, bias=False)

    def forward(self, x, t, cl, x_self_cond = None, mask=None):
        B, C, T = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.pos_embedding(pos)
        tok_emb = self.transformer.tok_embedding(x)

        time_emb = self.transformer.time_embedding(t)
        cl_emb = self.transformer.class_embedding(cl)

        x =  tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x, time_emb, cl_emb, mask)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x.permute(0, 2, 1) # (B, C, T)