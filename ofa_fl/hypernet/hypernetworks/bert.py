import copy
import random
from collections import OrderedDict
from typing import Optional, Tuple, Union, List
import os
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Embedding
from transformers.activations import ACT2FN
from transformers.adapters.composition import adjust_tensors_for_parallel, Stack, Fuse, Split, Parallel, BatchSplit
from transformers.adapters.lora import Linear as LoRALinear
from transformers.adapters.prefix_tuning import PrefixTuningShim
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, \
    BaseModelOutputWithPoolingAndCrossAttentions
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer, logger
from transformers.utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings

from hypernet.base.dynamic_modules import DynamicLinear, DynamicLayerNorm, MyIdentity
from hypernet.utils.evaluation import count_parameters

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import torch
from torch import nn, Tensor
from transformers.models.bert.modeling_bert import (
    BertForSequenceClassification,
    BertModel,
    BertOutput,
    BertIntermediate,
    BertSelfOutput,
    BertSelfAttention,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertEmbeddings,
    BertPooler, BERT_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC
)
from torchinfo import summary
import math
from transformers import BertModel, BertConfig, BertTokenizerFast, AutoAdapterModel, apply_chunking_to_forward, \
    AutoConfig, BertAdapterModel, AdapterSetup, ForwardContext

"""
    Code based on:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    
    The bert-model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
"""

class SuperBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = DynamicLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SuperBertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.dense =  DynamicLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DynamicLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()
        self.active_width_ratio = 1.0

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor,
                width_prune_ratio=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        prune_in_width = int(width_prune_ratio * self.config.intermediate_size)
        subnet = copy.deepcopy(self)
        subnet.active_width_ratio = width_prune_ratio
        subnet.dense = DynamicLinear(prune_in_width, self.config.hidden_size)
        subnet.dense.copy_linear(self.dense)

        return subnet

class SuperBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.dense = DynamicLinear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DynamicLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_adapter_modules()
        self.active_width_ratio = 1.0

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor,
                width_prune_ratio=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter_layer_forward(hidden_states, input_tensor, self.LayerNorm)
        return hidden_states

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        prune_width = int(width_prune_ratio * self.config.hidden_size)
        subnet = copy.deepcopy(self)
        subnet.active_width_ratio = width_prune_ratio
        subnet.dense = DynamicLinear(prune_width, self.config.hidden_size)
        subnet.dense.copy_linear(self.dense)
        return subnet

class SuperBertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, location_key: Optional[str] = None):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = DynamicLinear(config.hidden_size, self.all_head_size)
        self.key = DynamicLinear(config.hidden_size, self.all_head_size)
        self.value = DynamicLinear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        # 确保location_key被正确传递给PrefixTuningShim
        self.prefix_tuning = PrefixTuningShim(location_key, config)

        self.active_width_ratio = 1.0

    def transpose_for_scores(self, x: torch.Tensor, attention_head_size=None) -> torch.Tensor:
        if attention_head_size is None:
            attention_head_size = self.attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        width_prune_ratio=None,
    ) -> Tuple[torch.Tensor]:

        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio

        prune_hidden_size = int(width_prune_ratio * self.config.hidden_size)
        attention_head_size = prune_hidden_size // self.num_attention_heads
        all_head_size = self.num_attention_heads * attention_head_size
        mixed_query_layer = self.query(hidden_states, prune_hidden_size)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states, prune_hidden_size), attention_head_size)
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states, prune_hidden_size), attention_head_size)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states, prune_hidden_size), attention_head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states, prune_hidden_size), attention_head_size)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states, prune_hidden_size), attention_head_size)
            value_layer = self.transpose_for_scores(self.value(hidden_states, prune_hidden_size), attention_head_size)

        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        subnet = copy.deepcopy(self)
        subnet.active_width_ratio = width_prune_ratio
        prune_out_width = int(self.config.hidden_size * width_prune_ratio)
        subnet.query = DynamicLinear(self.config.hidden_size, prune_out_width)
        subnet.query.copy_linear(self.query)

        subnet.key = DynamicLinear(self.config.hidden_size, prune_out_width)
        subnet.key.copy_linear(self.key)

        subnet.value = DynamicLinear(self.config.hidden_size, prune_out_width)
        subnet.value.copy_linear(self.value)
        return subnet

class SuperBertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.config = config
        # 确保location_key被正确传递给SuperBertSelfAttention
        self.self = SuperBertSelfAttention(config, position_embedding_type=position_embedding_type, location_key="encoder")
        self.output = SuperBertSelfOutput(config)  # 动态宽度推理的输出层
        self.pruned_heads = set()
        self.active_width_ratio = 1.0  # 当前激活的宽度比例

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        width_prune_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor]:
        """
        动态宽度推理的前向传播。
        :param width_prune_ratio: 当前层的宽度比例，用于动态调整注意力机制的大小。
        """
        if width_prune_ratio is None:
            self.active_width_ratio = width_prune_ratio

        # 动态调整自注意力层的宽度
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            width_prune_ratio=width_prune_ratio,
        )
        attention_output = self_outputs[0]

        # 动态调整输出层的宽度
        attention_output = self.output(attention_output,
                                       hidden_states,
                                       width_prune_ratio=width_prune_ratio)

        outputs = (attention_output,) + self_outputs[1:]  # 添加注意力权重（如果需要）
        return outputs

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        subnet = copy.deepcopy(self)
        subnet.active_width_ratio = width_prune_ratio
        subnet.self = subnet.self.get_subnet(width_prune_ratio)
        subnet.output = subnet.output.get_subnet(width_prune_ratio)
        return subnet


class SuperBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = DynamicLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.active_width_ratio = 1.0

    def forward(self, hidden_states: torch.Tensor,
                width_prune_ratio=None) -> torch.Tensor:
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        width = int(self.config.intermediate_size * width_prune_ratio)
        hidden_states = self.dense(hidden_states, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio
        prune_out_width = int(self.config.intermediate_size * width_prune_ratio)
        subnet = copy.deepcopy(self)
        subnet.active_width_ratio = width_prune_ratio
        subnet.dense = DynamicLinear(self.config.hidden_size, prune_out_width)
        subnet.dense.copy_linear(self.dense)
        return subnet


class SuperBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = SuperBertAttention(config)  # 动态宽度推理的注意力层
        self.intermediate = SuperBertIntermediate(config)  # 动态宽度推理的中间层
        self.output = SuperBertOutput(config)  # 动态宽度推理的输出层
        self.active_width_ratio = 1.0  # 当前激活的宽度比例

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        width_prune_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor]:
        """
        动态宽度推理的前向传播。
        :param width_prune_ratio: 当前层的宽度比例，用于动态调整层的大小。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio

        # 动态调整注意力层的宽度
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            width_prune_ratio=width_prune_ratio,
        )
        attention_output = attention_outputs[0]

        # 动态调整中间层的宽度
        intermediate_output = self.intermediate(attention_output,
                                                width_prune_ratio=width_prune_ratio)

        # 动态调整输出层的宽度
        layer_output = self.output(intermediate_output,
                                   attention_output,
                                   width_prune_ratio=width_prune_ratio)

        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

    def get_subnet(self, width_prune_ratio: float = None):
        """
        提取当前层的子网，根据给定的宽度比例。
        :param width_prune_ratio: 子网的宽度比例。
        :return: 一个子网实例。
        """
        if width_prune_ratio is None:
            width_prune_ratio = self.active_width_ratio

        subnet = copy.deepcopy(self)
        subnet.attention = subnet.attention.get_subnet(width_prune_ratio)
        subnet.intermediate = subnet.intermediate.get_subnet(width_prune_ratio)
        subnet.output = subnet.output.get_subnet(width_prune_ratio)
        return subnet

class SuperBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([SuperBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        active_output_path=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        if active_output_path is None:
            active_output_path = self.config.active_output_path

        active_num_hidden_layers, active_width_prune_ratio = active_output_path

        for i in range(active_num_hidden_layers):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    active_width_prune_ratio
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    active_width_prune_ratio
                )

            hidden_states = layer_outputs[0]
            (attention_mask,) = adjust_tensors_for_parallel(hidden_states, attention_mask)

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def get_subnet(self, subnet_config):
        active_layers, active_width = subnet_config
        subnet = copy.deepcopy(self)
        subnet.config.active_output_path = subnet_config
        subnet.layer = subnet.layer[:active_layers]
        subnet.layer = nn.ModuleList([layer.get_subnet(active_width) for layer in subnet.layer])
        return subnet

class SuperBertModel(BertModel):

    def __init__(self, config):
        super(SuperBertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(self.config)

        self.encoder = SuperBertEncoder(config=self.config)

        self.pooler = SuperBertPooler(self.config)

        self._init_adapter_modules()

        # Initialize weights and apply final processing
        self.post_init()


    def load_pretrained_weights(self, pretrained_model_name="bert-base-uncased"):
        pretrained_model = BertModel.from_pretrained(pretrained_model_name)
        self.embeddings.load_state_dict(pretrained_model.embeddings.state_dict())
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self.query.load_state_dict(pretrained_model.encoder.layer[i].attention.self.query.state_dict())
            layer.attention.self.key.load_state_dict(pretrained_model.encoder.layer[i].attention.self.key.state_dict())
            layer.attention.self.value.load_state_dict(pretrained_model.encoder.layer[i].attention.self.value.state_dict())
            layer.attention.output.dense.load_state_dict(pretrained_model.encoder.layer[i].attention.output.dense.state_dict())
            layer.attention.output.LayerNorm.load_state_dict(pretrained_model.encoder.layer[i].attention.output.LayerNorm.state_dict())
            layer.intermediate.dense.load_state_dict(pretrained_model.encoder.layer[i].intermediate.dense.state_dict())
            layer.output.dense.load_state_dict(pretrained_model.encoder.layer[i].output.dense.state_dict())
            layer.output.LayerNorm.load_state_dict(pretrained_model.encoder.layer[i].output.LayerNorm.state_dict())
        self.pooler.load_state_dict(pretrained_model.pooler.state_dict())

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    @ForwardContext.wrap
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            active_output_path: Optional[float] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        embedding_output = self.invertible_adapters_forward(embedding_output)

        if active_output_path is None:
            active_output_path = self.config.active_output_path

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            active_output_path=active_output_path
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    def get_subnet(self, subnet_config=None):
        if subnet_config is None:
            subnet_config = self.active_output_path

        subnet = copy.deepcopy(self)
        subnet.config.active_output_path = subnet_config
        subnet.encoder = subnet.encoder.get_subnet(subnet_config)

        return subnet


class SuperBertAdapterModel(BertAdapterModel):


    def __init__(self, config: BertConfig):
        super().__init__(config)
        del self.bert
        self.config.default_output_path = (self.config.num_hidden_layers, 1.0)
        self.config.active_output_path = self.config.default_output_path
        self.bert = SuperBertModel(config)

        self._init_head_modules()

        self.init_weights()

        self.NUM_SUBLAYER = 1
        self.BASE_DEPTH_LIST = [self.config.num_hidden_layers]
        self.STAGE_WIDTH_LIST = [self.config.hidden_size]

        vec = torch.tensor(self.STAGE_WIDTH_LIST) / torch.sum(
            torch.tensor(self.BASE_DEPTH_LIST) * torch.tensor(self.STAGE_WIDTH_LIST))
        self.layer_cost = torch.repeat_interleave(vec, torch.tensor(self.BASE_DEPTH_LIST))

    def get_progressive_subnet_configs(self):

        subnetwork_configs = []

        active_layers, active_width = self.config.active_output_path
        width_pruning_ratio_list = self.config.width_pruning_ratio_list
        sublayer_ratio_list = width_pruning_ratio_list[width_pruning_ratio_list.index(active_width):]

        DepthProgressive = 0
        WidthProgressive = 1
        TwoDProgressive = 2
        RandProgressive = 3

        choices = [DepthProgressive, WidthProgressive, RandProgressive]

        chosen = random.choices(choices, k=1)[0]
        if chosen == DepthProgressive:
            subnetwork_configs = [(i+1, active_width) for i in range(active_layers)]

        elif chosen == WidthProgressive:
            subnetwork_configs = [(active_layers, width_ratio) for width_ratio in sublayer_ratio_list]

        else:
            for _ in range(5):
                subnetwork_config = self.random_sample_subnet_config()
                if subnetwork_config not in subnetwork_configs:
                    subnetwork_configs.append(subnetwork_config)
        random.shuffle(subnetwork_configs)
        subnetwork_configs = subnetwork_configs[:3]

        return subnetwork_configs

    def generate_all_subnet_configs(self):
        subnet_configs = []
        for i in range(self.config.num_hidden_layers):
            for k in self.config.width_pruning_ratio_list:
                subnet_configs.append((i+1, k))
        return subnet_configs

    def random_sample_subnet_config(self, max_net_config=None):
        if max_net_config is None:
            max_net_config = self.config.active_output_path
        width_pruning_ratio_list = self.config.width_pruning_ratio_list
        active_layers, active_width = max_net_config

        active_compression_rate = random.choice(width_pruning_ratio_list[width_pruning_ratio_list.index(active_width):])
        # 随机生成每个隐藏层的子层数量（至少为1）
        num_layers = random.randint(1, active_layers)

        return num_layers, active_compression_rate

    def load_pretrained_weights(self, pretrained_model_name_or_path: str):
        self.bert.load_pretrained_weights(pretrained_model_name_or_path)

    def get_subnet(self, subnet_config=None):
        if subnet_config is None:
            subnet_config = self.active_output_path

        subnet = copy.deepcopy(self)
        subnet.config.active_output_path = subnet_config
        subnet.bert = subnet.bert.get_subnet(subnet_config)
        return subnet

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            head=None,
            output_adapter_gating_scores=False,
            output_adapter_fusion_attentions=False,
            active_output_path=None,
            **kwargs
    ):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if active_output_path is None:
            active_output_path = self.config.active_output_path

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            active_output_path=active_output_path,
        )
        # BERT & RoBERTa return the pooled output as second item, we don't need that in these heads
        if not return_dict:
            head_inputs = (outputs[0],) + outputs[2:]
        else:
            head_inputs = outputs
        pooled_output = outputs[1]

        if head or AdapterSetup.get_context_head_setup() or self.active_head:
            head_outputs = self.forward_head(
                head_inputs,
                head_name=head,
                attention_mask=attention_mask,
                return_dict=return_dict,
                pooled_output=pooled_output,
                **kwargs,
            )
            return head_outputs
        else:
            # in case no head is used just return the output of the base model (including pooler output)
            return outputs

def super_bert_base(width_pruning_ratio_list: list, num_classes: int = 3) -> SuperBertAdapterModel:
    configuration = BertConfig()
    configuration.width_pruning_ratio_list = width_pruning_ratio_list
    configuration.num_classes = num_classes
    model = SuperBertAdapterModel(configuration)
    model.load_pretrained_weights("bert-base-uncased")
    model.add_adapter("mnli")
    model.add_classification_head("mnli", num_labels=num_classes)
    model.active_adapters = "mnli"
    model.train_adapter("mnli")
    return model

import matplotlib.pyplot as plt
import pandas as pd
from hypernet.utils.evaluation import visualize_model_parameters, count_parameters
from hypernet.datasets.load_mnli import get_datasets, collate_fn, load_mnli
def eval_all_subnets(model: SuperBertAdapterModel, data_path, device="cpu"):

    _, test_set = get_datasets(data_path)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4)

    def fine_tuning(m):
        m.train()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                labels = labels.to(device)
                outputs = m(inputs[0].to(device),
                            token_type_ids=inputs[1].to(device),
                            attention_mask=inputs[2].to(device),
                            output_hidden_states=True,)

    all_subnet_configs = model.generate_all_subnet_configs()
    all_subnet_configs.reverse()
    results = []
    for subnet_config in all_subnet_configs:
        print(subnet_config)
        subnet = model.get_subnet(subnet_config)
        # fine_tuning(subnet)
        model_size, accuracy = test(subnet, test_loader, device=device)
        results.append((model_size, accuracy))

    sizes, accuracies = zip(*results)
    pd.DataFrame({
        "size": sizes,
        "acc": accuracies,
        "subnet_configs": all_subnet_configs
    }).to_csv('bert_result.csv')

    plt.scatter(sizes, accuracies)
    plt.xlabel('Model Size (MB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Size vs Accuracy')
    plt.show()

def test(model, test_loader, active_path=None, device="cpu"):
    # 测试集测试准确率，召回率，F1，以及top-1,top-2,top-3
    correct = 0
    total = 0
    top_1 = 0
    top_2 = 0
    top_3 = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs[0].to(device),
                            token_type_ids=inputs[1].to(device),
                            attention_mask=inputs[2].to(device),
                            output_hidden_states=True,
                            active_output_path=active_path)
            outputs = outputs.logits.cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            _, top_2_predicted = torch.topk(outputs, 2)
            _, top_3_predicted = torch.topk(outputs, 3)
            for i in range(labels.size(0)):
                if labels[i] in top_2_predicted[i]:
                    top_2 += 1
                if labels[i] in top_3_predicted[i]:
                    top_3 += 1
                if labels[i] == predicted[i]:
                    top_1 += 1
    model_size, accuracy = count_parameters(model, 'MB'), 100 * correct / total
    print(active_path, f"Parameters:{model_size}MB")
    print(f"Accuracy: {accuracy}%")
    print(f"Top-1 Accuracy: {100 * top_1 / total}%")
    print(f"Top-2 Accuracy: {100 * top_2 / total}%")
    print(f"Top-3 Accuracy: {100 * top_3 / total}%")
    return model_size, accuracy

def train(model, train_loader, test_loader, active_config=None, sub_config=None, epochs=10, device="cpu"):
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-5, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # 迭代训练resnet到50个epoch，每10个epoch测试一次
    for epoch in range(epochs):
        model.train()
        # model.re_organize_weights()
        for i, (inputs, labels) in enumerate(train_loader):
            subnet_configs = model.get_progressive_subnet_configs()
            optimizer.zero_grad()
            outputs = model(inputs[0].to(device),
                            token_type_ids=inputs[1].to(device),
                            attention_mask=inputs[2].to(device),
                            output_hidden_states=True)
            loss = criterion(outputs.logits, labels.to(device, dtype=torch.long))
            n = 1
            for sub_config in subnet_configs:
                sub_outputs = model(inputs[0].to(device),
                                    token_type_ids=inputs[1].to(device),
                                    attention_mask=inputs[2].to(device),
                                    output_hidden_states=True,
                                    active_output_path=sub_config)
                sub_loss = criterion(sub_outputs.logits, labels.to(device, dtype=torch.long))
                loss += sub_loss
                n += 1
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print(f"Epoch {epoch} Iter {i} Loss {loss.item()}")
        if epoch % 5 == 0:
            test(model, test_loader, device=device)

def train_test(model: SuperBertAdapterModel, epochs=10, device="cpu"):
    data_shares = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.5]
    trainData, testData = load_mnli(os.path.join("../../../data", "mnli", 'original'),
                                    data_shares, 0.5, 1)

    train_loader = torch.utils.data.DataLoader(trainData[-1],
                                               batch_size=128,
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(testData[-1],
                                               batch_size=128,
                                               shuffle=False,
                                               collate_fn=collate_fn,
                                               num_workers=4)

    train(model, train_loader, test_loader, epochs=epochs, device=device)
    test(model, test_loader, device=device)
    for i in range(10):
        subnet_config = model.random_sample_subnet_config()
        test(model, test_loader, subnet_config, device=device)

        subnet = model.get_subnet(subnet_config)
        test(subnet, test_loader, device=device)


if __name__ == '__main__':
    # Initializing a model from the bert-base-uncased style configuration
    device = "cuda"
    bert = super_bert_base([1.0, 0.75, 0.5, 0.25], 3).to(device)
    train_test(bert, epochs=50, device=device)
    # bert = torch.load("../../logs/mnli/seed4321/hypernet.pth").to(device)
    # eval_all_subnets(bert, os.path.join("../../../data", "mnli", 'original'), device=device)



