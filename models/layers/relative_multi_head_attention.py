import torch
import torch.nn.functional as F
from multi_head_attention import AttentionHead, MultiHeadAttentionLayer
import misc

# TODO add residual connections
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

        # TODO remove unnecessary variables
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


class AdaptiveEmbedding(torch.nn.Module):
    """Implementation of 'Adaptive Input Representations for Neural Language Modeling'"""

    def __init__(
        self, num_tokens, dim_embed, dim_proj, cutoffs, div_val=1, sample_softmax=False
    ) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.dim_embed = dim_embed
        self.dim_proj = dim_proj
        self.cutoffs = cutoffs + [num_tokens]
        self.div_val = div_val
        self.sample_softmax = sample_softmax

        # derivations
        self.embedding_scale = self.dim_proj ** 0.5
        self.cutoff_ends = [0] + self.cutoffs

        # layers
        self.embedding_layers = torch.nn.ModuleList()
        self.embedding_projs = torch.nn.ParameterList()

        if self.div_val == 1:
            self.embedding_layers.append(
                torch.nn.Embedding(num_tokens, dim_embed, sparse=self.sample_softmax > 0)
            )

            if self.dim_proj != self.dim_embed:
                self.embedding_projs.append(
                    torch.nn.Parameter(torch.Tensor(self.dim_proj, self.dim_embed))
                )

        else:
            for i in range(len(self.cutoffs)):
                left_idx, right_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                dim_embed_ith = self.dim_embed // (self.div_val ** i)

                self.embedding_layers.append(
                    torch.nn.Embedding(right_idx - left_idx, dim_embed_ith)
                )
                self.embedding_projs.append(
                    torch.nn.Parameter(torch.Tensor(self.dim_proj, dim_embed_ith))
                )

    def forward(self, input_):
        """Forward method

        Args:
            input_ (torch.Tensor): input

        Returns:
            torch.Tensor: adaptive embedding of the word sequence
        """
        if self.div_val == 1:
            embed = self.embedding_layers[0](input_)
            if self.dim_proj != self.dim_embed:
                embed = F.linear(embed, self.embedding_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = input_.view(-1)
            emb_flat = torch.zeros(
                [inp_flat.size(0), self.dim_proj], dtype=param.dtype, device=param.device
            )
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.embedding_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.embedding_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*input_.size(), self.dim_proj)

        embed.mul_(self.embedding_scale)

        return embed


class TransformerXL(torch.nn.Module):
    def __init__(
        self,
        num_tokens,
        num_layers,
        num_heads,
        dim_model,
        dim_head,
        dim_inner,
        dropout,
        dim_embed=None,
        tie_weight=True,
        div_val=1,
        tie_projs=[False],
        pre_lnorm=False,
        tgt_len=None,
        ext_len=None,
        mem_len=None,
        cutoffs=None,
        adaptive_input=False,
        same_length=False,
        attention_type=0,
        clamp_len=-1,
        sample_softmax=-1,
    ) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_head = dim_head
        self.dim_inner = dim_inner
        self.dropout = dropout
        self.dim_embed = dim_model if dim_embed is None else dim_embed
        self.tie_weight = tie_weight
        self.div_val = div_val
        self.tie_projs = tie_projs
        self.pre_lnorm = pre_lnorm
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.cutoffs = cutoffs
        self.adaptive_input = adaptive_input
        self.same_length = same_length
        self.attention_type = attention_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax

        # derivations
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len

        # word embeddings
        self.word_embedding = AdaptiveEmbedding(
            num_tokens=self.num_tokens,
            dim_embed=self.dim_embed,
            dim_proj=self.dim_model,
            cutoffs=self.cutoffs,
            div_val=self.div_val,
        )

        # positional embeddings
        self.pos_embedding = misc.PositionalEmbedding(self.dim_model)

        # r_w_bias ==> bias_q
        # r_r_bias ==> bias_k
        self.bias_q = torch.nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.bias_k = torch.nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))

        # dropout
        self.dropout = torch.nn.Dropout(self.dropout)

        # decoders (only use 'attention_type' of '0')
        self.decoders = torch.nn.ModuleList(
            [
                RelativeAttentionDecoderLayer(
                    num_heads=self.num_heads,
                    dim_input=self.dim_model,
                    dim_head=self.dim_head,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # output layer
        self.out_layer = torch.nn.Linear(self.dim_model, self.num_tokens)

    def _reset_lengths(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len

    def _init_memory(self):
        memory = None

        if self.mem_len > 0:
            memory = []
            param = next(self.parameters())

            for _ in range(self.num_layers + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                memory.append(empty)

        return memory

    def _update_memory(self, hiddens, memory, qlen, mlen):
        # does not deal with None
        if memory is None:
            return None

        with torch.no_grad():
            new_memory = []
            end_idx = mlen + max(0, qlen - self.ext_len)
            start_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hiddens)):
                cat = torch.cat([memory[i], hiddens[i]], dim=0)
                new_memory.append(cat[start_idx:end_idx].detach())

        return new_memory

    def forward_(self, input_, memory=None):
        sequence_len, batch_size = input_.size()
        word_embedding = self.word_embedding(input_)
        pos_embedding = self.pos_embedding(sequence_len, clamp=self.clamp_len)

        memory_len = memory[0].size(0) if memory is not None else 0
        context_len = sequence_len + memory_len

        # TODO generate input masks

        # apply dropouts after embedding layers
        out = self.dropout(word_embedding)
        pos_embedding = self.dropout(pos_embedding)

        hiddens = [out]
        for idx, layer in enumerate(self.decoders):
            memory_ith = memory[idx] if memory is not None else None
            out = layer(out, pos_embedding, self.bias_q, self.bias_k, mems=memory_ith)
            hiddens.append(out)

        # apply dropout before return
        out = self.dropout(out)
        new_memory = self._update_memory(
            hiddens=hiddens, memory=memory, qlen=sequence_len, mlen=memory_len
        )

        return out, new_memory

    def forward(self, source, target, *memory):
        if not memory:
            memory = self._init_memory()

        tgt_len = target.size(0)
        hidden, new_memory = self.forward_(source, memory=memory)

        pred_hidden = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight

            logits = sample_logits(
                self.word_embedding, self.out_layer.bias, target, pred_hidden, self.sampler
            )

            loss = -F.log_softmax(logits, -1)[:, :, 0]

        else:
            loss = self.crit(pred_hidden.view(-1, pred_hidden.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_memory is None:
            return [loss]
        else:
            return [loss] + new_memory


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

        self.ff = misc.FeedForward(
            input_dim=self.dim_input, output_dim=self.dim_input, hidden_dim=self.dim_ff
        )

    def forward(self, word_embed, pos_embed, bias_q, bias_k, attention_mask=None, memories=None):
        out = self.attention(
            word_embed, pos_embed, bias_q, bias_k, attention_mask=attention_mask, memories=memories
        )

        out = self.ff(out)

        return out


if __name__ == "__main__":
    import misc

    print(misc.__dict__)
