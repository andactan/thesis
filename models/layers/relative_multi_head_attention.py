import torch
import torch.nn.functional as F
from multi_head_attention import AttentionHead, MultiHeadAttentionLayer
from misc import FeedForward


class RelativeMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_input: int,
        dim_head: int,
        dropout: float = 0.1,
        pre_lnorm: bool = False,
    ) -> None:
        """Relative Multi-Head Attention Layer from Transformer-XL paper

        Args:
            num_heads (int): number of heads
            input_dim (int): input embedding size
            query_dim (int): query size
            key_dim (int): key size
            value_dim (int): value size
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """

        super().__init__()

        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_head = dim_head
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm
        self.dim_out = self.num_heads * self.dim_head



        # query-key-value net
        self.qkv = torch.nn.Linear(self.dim_input, 3 * self.dim_out, bias=False)

        # linear layer
        self.linear = torch.nn.Linear(self.dim_out, self.dim_input, bias=False)

        # positional embedding net
        self.r_net = torch.nn.Linear(self.dim_input, self.dim_out)

        # misc
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(self.dim_input)

        # # relative positioning weights
        # self.pos_k = torch.nn.Linear(input_dim, key_dim, bias=False)
        # self.bias_k = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)
        # self.bias_q = torch.nn.Parameter(torch.Tensor(1, input_dim), requires_grad=True)

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        """Forward call

        Args:
            query (torch.Tensor): query input
            key (torch.Tensor): key input
            value (torch.Tensor): value input
            pos_embed (torch.Tensor): positional embeddings
        """

        # r_w_bias ==> bias_q
        # r_r_bias ==> bias_k

        query_len, pos_len, batch_size = word_embed.size(0), pos_embed.size(0), word_embed.size(1)

        # append memories if not None
        input_ = word_embed
        if memories is not None:
            input_ = torch.cat([memories, word_embed], 0)

        if self.pre_lnorm:
            input_ = self.pre_lnorm(input_)

        qkv = self.qkv(input_)

        query_, key_, value_ = torch.chunk(qkv, 3, dim=-1)
        if memories is not None:
            query_ = query_[-query_len:]  # get activations only for sequence, freeze others

        # shift the positonal embeddings
        shifted_pos_embed = self._relative_shift(pos_embed)
        pos_key_ = self.r_net(shifted_pos_embed)

        q_bias_q = query_ + bias_q
        AC = q_bias_q.bmm(key_.transpose(1, 2))

        k_bias_k = key_ + bias_k
        BD = k_bias_k.bmm(pos_key_.transpose(1, 2))

        # calculate attention
        pre_attn = AC + BD
        scale = key_.size(-1) ** 0.5
        attn = F.softmax(pre_attn / scale, dim=-1)
        attn = attn.bmm(value_)
        out = self.linear(attn)
        # out = self.dropout(out)

        return out

    def _relative_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x


class RelativeAttentionDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_input: int,
        dim_head: int,
        dim_ff: int,
        dropout: float = 0.1,
        pre_lnorm: bool = False,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.attention = RelativeMultiHeadAttention(
            num_heads=self.num_heads,
            dim_input=self.dim_input,
            dim_head=self.dim_head,
            dropout=self.dropout,
            pre_lnorm=self.pre_lnorm,
        )

        self.ff = FeedForward(
            input_dim=self.dim_input, output_dim=self.dim_input, hidden_dim=self.dim_ff
        )

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        out = self.attention(
            word_embed, pos_embed, bias_q, bias_k, attention_mask=attention_mask, memories=memories
        )

        out = self.ff(out)

        return out

# TODO: compare original implementation and refactored one

import random
import numpy as np

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

num_heads = 2
dim_input = 200
dim_head = 2

word_embed = torch.ones(36, 4, 200)
pos_embed = torch.ones(36, 1, 200)
bias_q = torch.ones(1, num_heads * dim_head)
bias_k = torch.ones(1, num_heads * dim_head)

import torch.nn as nn

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        print(qlen)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:] # get activations only for sequence, freeze others
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        # attn_out = self.drop(attn_out)

        # if self.pre_lnorm:
        #     ##### residual connection
        #     output = w + attn_out
        # else:
        #     ##### residual connection + layer normalization
        #     output = self.layer_norm(w + attn_out)

        return attn_out


m = RelativeMultiHeadAttention(
    num_heads=num_heads,
    dim_input=dim_input,
    dim_head=dim_head
)

print(m(word_embed, pos_embed, bias_q, bias_k))


bias_q = torch.ones(num_heads, dim_head)
bias_k = torch.ones(num_heads, dim_head)
m2 = RelPartialLearnableMultiHeadAttn(
    n_head=num_heads,
    d_model=dim_input,
    d_head=dim_head,
    dropout=0.1
)

# for name, param in m2.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
m2.qkv_net.weight.data = m.qkv.weight.data
m2.o_net.weight.data = m.linear.weight.data
m2.r_net.weight.data = m.r_net.weight.data
print(m2(word_embed, pos_embed, bias_q, bias_k))