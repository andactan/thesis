import torch
import math
import torch.nn.functional as F

from inspect import isfunction
from functools import partial
from mogrifier import Mogrifier
from collections import namedtuple
from torch.nn.modules.linear import Linear


Memory = namedtuple("Memory", ["mem", "compressed_mem"])

# helper functions
def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def to(t):
    return {"dtype": t.dtype, "device": t.device}


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim : dim + 1] = split_dims
    return t.reshape(shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1 :]


# full attention for calculating auxiliary reconstruction loss
def full_attn(q, k, v, dropout_fn=None):
    *_, dim = q.shape
    dots = torch.einsum("bhid,bhjd->bhij", q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum("bhij,bhjd->bhid", attn, v)

# tensor iterator
def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]


# helper classes
class Residual(torch.nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class GRUTypeGate(torch.nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.w_r = Linear(input_size, input_size, bias=False)
        self.u_r = Linear(input_size, input_size, bias=False)
        self.w_z = Linear(input_size, input_size, bias=False)
        self.u_z = Linear(input_size, input_size, bias=False)
        self.w_g = Linear(input_size, input_size, bias=False)
        self.u_g = Linear(input_size, input_size, bias=False)
        self.register_parameter(name="b_g", param=torch.nn.Parameter(torch.ones(input_size) * 2))

    def forward(self, x, y):
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        z = torch.sigmoid(self.w_z(y) + self.u_z(x) - self.b_g)
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))
        g = (1 - z) * x + z * h
        return g


class GRUGating(torch.nn.Module):
    def __init__(self, dim, fn, mogrify=False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        # self.gru = nn.GRUCell(dim, dim)
        self.mogrify = Mogrifier(dim, factorize_k=dim // 4) if mogrify else None
        self.gate = GRUTypeGate(dim)

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        gated_output = self.gate(x, torch.relu(y))
        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest

        return ret


class ConvCompress(torch.nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, ratio, stride=ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


class CustomGELU(torch.nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = torch.nn.GELU if hasattr(torch.nn, "GELU") else CustomGELU


class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = torch.nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = torch.nn.Dropout(dropout)
        self.w2 = torch.nn.Linear(dim * mult, dim)

    #TODO: remove kwargs
    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        sequence_len,
        memory_len,
        cmem_len,
        cmem_ratio=4,
        heads=8,
        attention_dropout=0.0,
        dropout=0.0,
        reconstruction_attention_dropout=0,
        one_kv_head=False,
    ) -> None:
        super().__init__()

        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"

        # variables
        self.heads = heads
        self.dim = dim
        self.dim_head = dim // heads
        self.sequence_len = sequence_len
        self.memory_len = memory_len
        self.cmem_len = cmem_len
        self.cmem_ratio = cmem_ratio
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.reconstruction_attention_dropout = reconstruction_attention_dropout
        self.one_kv_head = one_kv_head
        self.scale = self.dim_head ** (-0.5)

        # compress memory function
        self.compress_memory_fn = ConvCompress(self.dim, self.cmem_ratio)

        # to query
        self.to_q = torch.nn.Linear(self.dim, self.dim, bias=False)

        # KV (key-value) head
        kv_dim = self.dim_head if self.one_kv_head else self.dim
        self.to_kv = torch.nn.Linear(self.dim, kv_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(self.dim, self.dim)

        # dropout
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        # attention dropout
        self.attention_dropout_layer = torch.nn.Dropout(self.attention_dropout)

        # reconstruction attention dropout
        self.reconstruction_attention_dropout_layer = torch.nn.Dropout(
            self.reconstruction_attention_dropout
        )

        # layer normalization
        self.layer_norm = torch.nn.LayerNorm(self.dim)

    def forward(self, x, memories=None, pos_embed=None, input_mask=None, calc_memory=None):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        # memory
        memories = default(memories, (None, None))
        mem, cmem = memories
        init_empty_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        mem_len = mem.shape[1]
        cmem_len = cmem.shape[1]

        # query
        q = self.to_q(x)

        kv_input = torch.cat((cmem, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))
        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_embed is not None:
            pos_embed = pos_embed[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum("bhid,hjd->bhij", q, pos_embed) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, (mem_len + cmem_len, 0), value=True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + cmem_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal=1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attention_dropout_layer(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout_layer(logits)

        logits = self.layer_norm(x + logits)

        new_mem = mem
        new_cmem = cmem
        aux_loss = torch.zeros(1, requires_grad=True, **to(q))

        if self.sequence_len > t or not calc_memory:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = split_at_index(1, -self.memory_len, torch.cat((mem, x), dim=1))
        old_mem_padding = old_mem.shape[1] % self.cmem_ratio

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, (0, 0, old_mem_padding, 0), value=0.0)

        if old_mem.shape[1] == 0 or self.cmem_len <= 0:
            return logits, Memory(new_mem, new_cmem), aux_loss

        compressed_mem = self.compress_memory_fn(old_mem)
        old_cmem, new_cmem = split_at_index(
            1, -self.cmem_len, torch.cat((cmem, compressed_mem), dim=1)
        )

        if not self.training:
            return logits, Memory(new_mem, new_cmem), aux_loss

        # calculate compressed memory auxiliary loss if training

        cmem_k, cmem_v = self.to_kv(compressed_mem).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, h, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(
            -min(mem_len, self.memory_len) - self.sequence_len, -self.sequence_len
        )
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        q, old_mem_k, old_mem_v, cmem_k, cmem_v = map(
            torch.detach, (q, old_mem_k, old_mem_v, cmem_k, cmem_v)
        )

        attn_fn = partial(full_attn, dropout_fn=self.reconstruction_attention_dropout_layer)

        aux_loss = F.mse_loss(attn_fn(q, old_mem_k, old_mem_v), attn_fn(q, cmem_k, cmem_v))

        return logits, Memory(new_mem, new_cmem), aux_loss


class CompressiveTransformer(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        sequence_len,
        depth,
        embedding_dim=None,
        memory_layers=None,
        memory_len=None,
        cmem_len=None,
        cmem_ratio=4,
        heads=8,
        gru_gated_residual=True,
        mogrify_gru=False,
        attention_dropout=0.8,
        ff_glu=False,
        ff_dropout=0.0,
        dropout=0.0,
        reconstruction_attention_dropout=0.0,
        reconstruction_loss_weight=1.0,
        one_kv_head=False,
    ) -> None:
        super().__init__()

        self.embedding_dim = default(embedding_dim, dim)
        self.memory_len = default(memory_len, sequence_len)
        self.cmem_len = default(cmem_len, memory_len // cmem_ratio)
        self._memory_layers = default(memory_layers, list(range(1, depth + 1)))

        # variable initializations
        self.sequence_len = sequence_len
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.depth = depth
        self.memory_layers = list(self._memory_layers)
        self.token_embedding = torch.nn.Embedding(num_tokens, embedding_dim)
        self.to_model_dim = (
            torch.nn.Identity()
            if self.embedding_dim == dim
            else torch.nn.Linear(self.embedding_dim, dim)
        )

        # positional embedding
        sequence_mem_len = self.sequence_len + self.memory_len + self.cmem_len
        self.pos_embedding = torch.nn.Parameter(torch.zeros(heads, sequence_mem_len, dim // heads))

        # to logits
        self.to_logits = torch.nn.Sequential(
            torch.nn.Identity()
            if self.embedding_dim == dim
            else torch.nn.Linear(dim, self.embedding_dim),
            torch.nn.Linear(self.embedding_dim, num_tokens),
        )

        # attention layers
        wrapper = partial(GRUGating, dim, mogrify=mogrify_gru) if gru_gated_residual else Residual

        #! refactor and separate norm and attention layers
        #! change here for Metaformer
        self.attention_layers = torch.nn.ModuleList(
            [
                wrapper(
                    PreNorm(
                        dim,
                        SelfAttention(
                            dim,
                            sequence_len,
                            self.memory_len,
                            self.cmem_len,
                            cmem_ratio,
                            heads,
                            dropout=dropout,
                            attention_dropout=attention_dropout,
                            reconstruction_attention_dropout=reconstruction_attention_dropout,
                            one_kv_head=one_kv_head,
                        ),
                    )
                )
                for _ in range(depth)
            ]
        )

        #! refactor and separate norm and FF layers
        # feed forward layers
        self.ff_layers = torch.nn.ModuleList(
            [
                wrapper(
                    PreNorm(
                        dim,
                        FeedForward(dim, dropout=ff_dropout, glu=ff_glu, activation=torch.nn.ReLU),
                    )
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, memories=None, mask=None):
        x = self.token_embedding(x)
        x = self.to_model_dim(x)

        b, t, d = x.shape

        assert (
            t <= self.sequence_len
        ), f"input contains a sequence length {t} that is greater than the designated maximum sequence length {self.sequence_len}"


        memories = default(memories, (None, None))
        mem, cmem = memories

        num_memory_layers = len(self.memory_layers)
        init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, **to(x))
        mem = default(mem, init_empty_mem)
        cmem = default(cmem, init_empty_mem)

        total_len = mem.shape[2] + cmem.shape[2] + self.sequence_len
        pos_emb = self.pos_embedding[:, (self.sequence_len - t):total_len]

        next_mem = []
        next_cmem = []
        aux_loss = torch.tensor(0., requires_grad=True, **to(x))

        # create an iterator to iterate through memories
        mem_iter, cmem_iter = map(iterate_tensor, (mem, cmem))

        for ind, (attn, ff) in enumerate(zip(self.attention_layers, self.ff_layers)):
            layer_num = ind + 1

            use_memory = layer_num in self.memory_layers
            memories = (next(mem_iter), next(cmem_iter)) if use_memory else None

            x, (mem_out, cmem_out), layer_aux_loss = attn(x, memories=memories, calc_memory=use_memory, input_mask=mask,
                                                          pos_embed=pos_emb)
            x, = ff(x)

            aux_loss = aux_loss + layer_aux_loss

            if not use_memory:
                continue

            next_mem.append(mem_out)
            next_cmem.append(cmem_out)

        out = self.to_logits(x)

        next_mem, next_cmem = map(torch.stack, (next_mem, next_cmem))
        next_mem, next_cmem = map(torch.detach, (next_mem, next_cmem))

        aux_loss = aux_loss * self.reconstruction_loss_weight / num_memory_layers
        return out, Memory(mem=next_mem, compressed_mem=next_cmem), aux_loss
