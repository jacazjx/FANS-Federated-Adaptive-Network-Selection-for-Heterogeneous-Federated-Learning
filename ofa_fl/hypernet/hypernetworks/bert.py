from typing import Optional, Tuple
import os
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, Embedding

from hypernet.base.dynamic_modules import DynamicLinear

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
    BertPooler
)
from torchinfo import summary
import math

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




class SuperBertModel(nn.Module):

    def __init__(self):
        self.embeddings = None
        self.encoder = None
        self.classifier = None

        self.load_pretrained()

    def load_pretrained(self):
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        self.embeddings = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.classifier = nn.ModuleList()

        for n, m in model.named_modules():
            for np, p in m.named_parameters(recurse=False):
                print(f"{n}.{np}")


    @staticmethod
    def replace_layer(model):
        """
        Replace layers of type `old_layer_type` in the model with the given `new_layer`.

        Args:
            model (nn.Module): The original model.
            old_layer_type (type): The type of layer you want to replace, e.g., nn.Linear.
            new_layer (callable): A function or class that returns a new layer instance when called.
        """
        for name, module in model.named_children():
            if isinstance(module, Linear):
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias
                dy_module = DynamicLinear(in_features, out_features, bias)
                dy_module.linear = module
                setattr(model, name, dy_module)


            elif isinstance(module, LayerNorm):
                in_features = module.in_features
                out_features = module.out_features
                bias = module.bias is not None
                # Replace with the new custom layer
                setattr(model, name, DynamicLayerNorm(in_features, out_features, bias))
            else:
                # Recursively apply to child modules (e.g., in nested structures like nn.Sequential)
                replace_layer(module, old_layer_type, new_layer)




if __name__ == '__main__':
    # from transformers import BertModel
    #
    # model = BertModel.from_pretrained("bert-base-uncased")
    # print(model)
    from transformers import BertModel,   BertConfig, BertTokenizerFast, AutoModel
    from adapters import AutoAdapterModel

    device = "cuda"
    # Initializing a BERT bert-base-uncased style configuration
    configuration = BertConfig()
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    p_id = np.random.randint(100, 10000, 10).tolist()
    h_id = np.random.randint(100, 10000, 10).tolist()
    pair_token_ids = torch.LongTensor([[tokenizer.cls_token_id] + p_id + [
        tokenizer.sep_token_id] + h_id + [tokenizer.sep_token_id]]).to(device)
    segment_ids = torch.LongTensor([[0] * (len(p_id) + 2) + [1] * (len(h_id) + 1)]).to(device)
    attention_mask_ids = torch.LongTensor([[1] * (len(p_id) + len(h_id) + 3)]).to(device)
    dummy_input = (pair_token_ids, segment_ids, attention_mask_ids)

    # Initializing a model from the bert-base-uncased style configuration
    model = BertModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    print(configuration)
    print(model)
    modules = {}
    for n, m in model.named_modules():
        for np, p in m.named_parameters(recurse=False):
            print(f"{n}.{np}")
