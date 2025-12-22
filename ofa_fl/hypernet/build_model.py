import numpy as np
import torch
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import BertTokenizerFast
from .hypernetworks import resnet, bert, densenet


# trs = False means use static BatchNorm

def build_model(width_ratio_list, task, n_class, device):
    if task.startswith('cifar'):
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
    elif task == 'mnist':
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
    elif task == 'mnli':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        p_id = np.random.randint(100, 10000, 10).tolist()
        h_id = np.random.randint(100, 10000, 10).tolist()
        pair_token_ids = torch.LongTensor([[tokenizer.cls_token_id] + p_id + [
            tokenizer.sep_token_id] + h_id + [tokenizer.sep_token_id]]).to(device)
        segment_ids = torch.LongTensor([[0] * (len(p_id) + 2) + [1] * (len(h_id) + 1)]).to(device)
        attention_mask_ids = torch.LongTensor([[1] * (len(p_id) + len(h_id) + 3)]).to(device)
        dummy_input = (pair_token_ids, segment_ids, attention_mask_ids)
    elif task == 'imagenet':
        dummy_input = torch.randn(1, 3, 64, 64, device=device)

    if task == "cifar10":
        model = resnet.super_resnet18(True, False, width_ratio_list, num_classes=n_class)

    elif task == "cifar100":
        model = densenet.super_densenet121(width_pruning_ratio_list=width_ratio_list, num_classes=n_class)

    elif task == "imagenet":
        model = resnet.super_resnet18(True, False, width_ratio_list, num_classes=n_class, input_size=(1, 3, 64, 64))

    else:
        model = bert.super_bert_base(width_pruning_ratio_list=width_ratio_list, num_classes=n_class)

    model.dummy_input = dummy_input

    return model



