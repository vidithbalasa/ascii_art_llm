import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import wraps
from einops import rearrange, repeat
from timm.models.layers import DropPath

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        # Ensure the normalized shape matches the feature dimension of the input
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, drop_path_rate=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.drop_path = nn.Identity()  # Simplified for clarity

    def forward(self, x):
        original_dim = x.dim()  # Get the original number of dimensions
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Temporarily add a sequence dimension if it's 2D

        b, n, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=8), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if original_dim == 2:
            out = out.squeeze(1)  # Remove the temporary sequence dimension if originally 2D

        return self.to_out(self.drop_path(out))

class ASCIITransformer(nn.Module):
    def __init__(self, depth=8, queries_dim=384, hidden_dim=256, output_dim=32*32*96, dim=384, dim_head=32, heads=8, decoder_ff=False):
        super(ASCIITransformer, self).__init__()
        self.output_dim = output_dim  # Total output dimension

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(),
                get_latent_ff()
            ]))

        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_outputs = nn.Linear(queries_dim, self.output_dim)  # Ensure this is connected to the output_dim correctly

    def decode(self, embedding):
        for self_attn, self_ff in self.layers:
            embedding = self_attn(embedding) + embedding
            embedding = self_ff(embedding) + embedding

        if self.decoder_ff is not None:
            embedding = self.decoder_ff(embedding)
        return self.to_outputs(embedding)

    def forward(self, x):
        o = self.decode(x)
        o = o.view(-1, 1024, 96)  # Reshape output to [batch_size, 1024, 96]
        predicted_indices = torch.argmax(o, dim=2)  # Apply argmax to convert logits to class indices
        return predicted_indices

class ASCIITokenizer:
    def __init__(self):
        self.vocab = [chr(i) for i in range(32, 127)] + ['']
        self.vocab_size = len(self.vocab)
        self.token_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.char_to_token = {c: i for i, c in enumerate(self.vocab)}

    def tokenize(self, text):
        tokens = [self.char_to_token[c] for c in text if c in self.vocab]
        return tokens