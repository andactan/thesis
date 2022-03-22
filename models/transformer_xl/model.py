import torch
import torch.nn.functional as F

from models.transformer_xl.misc import AdaptiveEmbedding, PositionalEmbedding
from models.transformer_xl.decoder import RelativeMultiHeadAttentionDecoderLayer
from models.transformer_xl.utils import sample_logits, LogUniformSampler
from models.transformer_xl.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


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
        self.pos_embedding = PositionalEmbedding(self.dim_model)

        # r_w_bias ==> bias_q
        # r_r_bias ==> bias_k
        self.bias_q = torch.nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))
        self.bias_k = torch.nn.Parameter(torch.Tensor(self.num_heads, self.dim_head))

        # dropout
        self.dropout = torch.nn.Dropout(self.dropout)

        # decoders (only use 'attention_type' of '0')
        self.decoders = torch.nn.ModuleList(
            [
                RelativeMultiHeadAttentionDecoderLayer(
                    num_heads=self.num_heads,
                    dim_input=self.dim_model,
                    dim_head=self.dim_head,
                    dropout=self.dropout,
                )
                for _ in range(self.num_layers)
            ]
        )

        # use sampled softmax
        if self.sample_softmax > 0:
            # output layer
            self.out_layer = torch.nn.Linear(self.dim_model, self.num_tokens)

            if self.tie_weight:
                self.out_layer.weight = self.word_embedding.weight

            self.sampler = LogUniformSampler(self.num_tokens, self.sample_softmax)

        # use adaptive softmax
        else:
            self.criterion = ProjectedAdaptiveLogSoftmax(
                self.num_tokens, self.dim_embed, self.dim_model, self.cutoffs, self.div_val
            )

            if self.tie_weight:
                for i in range(len(self.criterion.out_layers)):
                    self.criterion.out_layers[i].weight = self.word_embedding.embedding_layers[
                        i
                    ].weight

            if self.tie_projs:
                for i, tie_proj in enumerate(self.tie_projs):
                    if tie_proj and self.div_val == 1 and (self.dim_model != self.dim_embed):
                        self.criterion.out_projs[i] = self.word_embedding.embedding_projs[0]
                    elif tie_proj and self.div_val != 1:
                        self.criterion.out_projs[i] = self.word_embedding.embedding_projs[i]

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
            loss = self.criterion(pred_hidden.view(-1, pred_hidden.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_memory is None:
            return [loss]
        else:
            return [loss] + new_memory
