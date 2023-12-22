from typing import Optional, Tuple, Union
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2Block,GPT2Attention
from transformers import GPT2PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,CausalLMOutputWithCrossAttentions)
from transformers.pytorch_utils import Conv1D
from transformers.activations import ACT2FN
from transformers.utils import logging
from protein_lm.modeling.utils.rotary_embedding import RotaryEmbedding
from protein_lm.modeling.utils.rerope_embedding import RectifiedRotaryEmbedding
from protein_lm.modeling.utils.alibi_embedding import create_alibi_tensor
from protein_lm.modeling.utils.scaled_rope_embedding import LlamaLinearScalingRotaryEmbedding,LlamaDynamicNTKScalingRotaryEmbedding
from protein_lm.modeling.utils.modules import ContactPredictionHead

logger = logging.get_logger(__name__)

class APTAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool)).view(
                1, 1, self.max_positions, self.max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.position_embedding = config.position_embedding
        self.rope_scaling_factor=config.rope_scaling_factor
        self.rope_theta=config.rope_theta
        self.max_sequence_length = config.max_sequence_length
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        # muP
        self.use_mup = config.use_mup

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)

            #muP -- q_attn
            if self.use_mup:
                self.q_attn.weight.data.normal_(mean=0.0, std=math.sqrt((config.initializer_range ** 2) / config.mup_width_scale))
                self.q_attn.bias.zero_()

        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)

        #muP -- c_attn specific, see https://github.com/cofe-ai/Mu-scaling/blob/a3d13c3eaa02e28cc6ef95116358b128424f9fe4/modeling/modeling_gpt2_mup.py#L487
        if self.use_mup:
            if config.query_zero_init:
                _, fanout = self.c_attn.weight.shape
                self.c_attn.weight.data[:, :fanout//3] = 0
                self.c_attn.bias.zero_()

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        
        #muP -- c_proj specific, see https://github.com/cofe-ai/Mu-scaling/blob/a3d13c3eaa02e28cc6ef95116358b128424f9fe4/modeling/modeling_gpt2_mup.py#L494
        if self.use_mup:
            depth_std = config.initializer_range / math.sqrt(2 * config.n_layer)
            self.c_proj.weight.data.normal_(mean=0.0, std=math.sqrt(config.depth_std ** 2 / config.mup_width_scale))
            self.c_proj.bias.zero_()
        
        if self.use_mup:
            self.attn_dropout = nn.Identity()
            self.resid_dropout = nn.Identity()
        else:
            self.attn_dropout = nn.Dropout(config.attn_pdrop)
            self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        

        self.rot_emb=None
        if self.position_embedding == "rope":
            self.rot_emb=RotaryEmbedding(dim=self.head_dim)
        elif self.position_embedding == "rerope":
            self.rot_emb = RectifiedRotaryEmbedding(dim=self.head_dim,max_position_embeddings = self.max_positions)
        elif self.position_embedding=="linear_rope_scaling":
            self.rot_emb=LlamaLinearScalingRotaryEmbedding(dim=self.head_dim,max_position_embeddings=self.max_positions,scaling_factor=self.rope_scaling_factor,base=self.rope_theta)
        elif self.position_embedding=="dynamic_rope_scaling":
            self.rot_emb=LlamaDynamicNTKScalingRotaryEmbedding(dim=self.head_dim,max_position_embeddings=self.max_positions,scaling_factor=self.rope_scaling_factor,base=self.rope_theta)

    

    def _attn(self, query, key, value, attention_mask=None, head_mask=None,alibi_bias=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        #muP
        if self.use_mup:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1), dtype=attn_weights.dtype, device=attn_weights.device
            )
        elif self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        if alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias[:,:,:attn_weights.size(-1)]
        
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None,alibi_bias=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias[:,:,:attn_weights.size(-1)]
        
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        alibi_bias: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,

    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `APTAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        kv_seq_len=key.shape[-2]
        if layer_past is not None:
            kv_seq_len+=layer_past[0].shape[-2]
      
        # Apply rope embedding to query and key
        if self.rot_emb:
            bsz, q_len, _ = hidden_states.size()
            if self.position_embedding == 'rope':
                query, key = self.rot_emb(query,key)
            elif self.position_embedding == 'rerope':
                query = query.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                query *= ((position_ids + 1)[:, None, :, None].log() / torch.log(torch.tensor(self.max_sequence_length)).item()).clip(1).to(query.dtype)
                query, key = self.rot_emb(query,key,seq_len = self.max_sequence_length,position_ids=position_ids)
            elif self.position_embedding=="linear_rope_scaling" or self.position_embedding=="dynamic_rope_scaling":
                query = query.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                query, key = self.rot_emb(query, key, seq_len=kv_seq_len,position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)


        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask,alibi_bias=alibi_bias)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask,alibi_bias=alibi_bias)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class APTMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size

        #muP
        use_mup = config.use_mup

        self.c_fc = Conv1D(intermediate_size, embed_dim)

        #muP -- matrix-like
        if use_mup:
            self.c_fc.weight.data.normal_(mean=0.0, std=math.sqrt((config.initializer_range ** 2) / config.mup_width_scale))
            self.c_fc.bias.zero_()

        self.c_proj = Conv1D(embed_dim, intermediate_size)

        #muP -- matrix-like, c_proj-specific, see https://github.com/cofe-ai/Mu-scaling/blob/a3d13c3eaa02e28cc6ef95116358b128424f9fe4/modeling/modeling_gpt2_mup.py#L494
        if use_mup:
            depth_std = config.initializer_range / math.sqrt(2 * config.n_layer)
            self.c_proj.weight.data.normal_(mean=0.0, std=math.sqrt(depth_std ** 2 / config.mup_width_scale))
            self.c_proj.bias.zero_()

        self.act = ACT2FN[config.activation_function]

        if use_mup:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class APTBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        #muP
        use_mup = config.use_mup

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        #muP -- vector-like
        if self.use_mup:
            self.ln_1.weight.data.fill_(1.0)
            self.ln_1.bias.data.zero_()
        
        self.attn = APTAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        #muP -- vector-like
        if use_mup:
            self.ln_2.weight.data.fill_(1.0)
            self.ln_2.bias.data.zero_()

        if config.add_cross_attention:
            #muP TO DO: check proper behavior in case of crossattention
            self.crossattention = APTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

            #muP -- vector-like
            if use_mup:
                self.ln_cross_attn.weight.data.fill_(1.0)
                self.ln_cross_attn.bias.data.zero_()

        self.mlp = APTMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        alibi_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi_bias=alibi_bias,
            position_ids=position_ids,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)



"""The bare APT Model transformer outputting raw hidden-states without any specific head on top."""
class APTModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_sizeù
        use_mup = config.use_mup

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        
        #muP -- vector-like, zero if zero init or mantained regardless of width
        if use_mup:
            if config.wte_zero_init:
                self.wte.weight.data.zero_()
            else:
                self.wte.weight.data.normal_(mean=0.0, std=config.initializer_range)

            if self.wte.padding_idx is not None:
                self.wte.weight.data[self.wte.padding_idx].zero_()

        self.position_embedding = config.position_embedding if hasattr(config, "position_embedding") else "learned"
        
        if self.position_embedding=="learned" or self.position_embedding == 'rope' or self.position_embedding == 'rerope' or self.position_embedding=="linear_rope_scaling" or self.position_embedding =="dynamic_rope_scaling":
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
            
            #muP -- vector-like, constant regardless of width
            #muP TO DO: check proper behavior in rope & rerope case
            if self.use_mup:
                self.wpe.weight.data.normal_(0.0, std=config.initializer_range)

                if self.wpe.padding_idx is not None:
                    self.wpe.weight.data[self.wte.padding_idx].zero_()

            self.alibi = None
        elif self.position_embedding=="alibi":
            #muP TO DO: check proper behavior in alibi case 
            
            maxpos = config.n_positions
            attn_heads = config.n_head
            alibi = create_alibi_tensor(attn_heads,maxpos)
            self.register_buffer('alibi',alibi)
        else:
            raise Exception(f'position_embedding {self.position_embedding} not supported. Please select one of learned, rope, rerope, linear rope, dynamic rope or alibi')
        
        #muP
        if use_mup:
            self.drop = nn.Identity()
        else:
            self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([APTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        #muP -- vector-like
        if use_mup:
            self.ln_f.weight.data.fill_(1.0)
            self.ln_f.bias.data.zero_()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.position_embedding=="learned" or self.position_embedding == 'rope' or self.position_embedding == 'rerope' or self.position_embedding=="linear_rope_scaling" or self.position_embedding =="dynamic_rope_scaling":
            position_embeds = self.wpe(position_ids)
            position_embeds.mul_(self.mup_rp_embedding_mult)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi_bias=self.alibi if hasattr(self, "alibi") else None,
                    position_ids=position_ids
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    
"""
The APT Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).
"""
class APTLMHeadModel(GPT2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = APTModel(config)

        #muP TO DO: check proper behavior for LM head, nothing should be done (?)
        #see https://github.com/cofe-ai/Mu-scaling/blob/a3d13c3eaa02e28cc6ef95116358b128424f9fe4/modeling/modeling_gpt2_mup.py#L472
        #see also table 8's caption in https://arxiv.org/pdf/2203.03466.pdf
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
        self.contact_head=ContactPredictionHead(config.num_hidden_layers * config.num_attention_heads,
                                            prepend_bos=True,
                                            append_eos=True,
                                            eos_idx=2)        

        # Initialize weights and apply final processing
        self.post_init()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def predict_contacts(self, input_ids):
        transformer_outputs = self.transformer(
            input_ids,
            return_dict=True,  
            output_attentions=True,  
        )
        # Convert attention tuples to list
        attentions_list = list(transformer_outputs.attentions)

        # Stack the attention tensors
        stacked_attentions = torch.stack(
            [attn for attn in attentions_list],
            dim=1
        )
               
        contact_predictions = self.contact_head(input_ids, stacked_attentions)

        return contact_predictions