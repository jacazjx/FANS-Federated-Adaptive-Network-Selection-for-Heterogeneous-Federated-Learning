import numpy as np
import torch
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import BertTokenizerFast, AutoAdapterModel
from .hypernetworks import resnet, bert, vgg, densenet


# trs = False means use static BatchNorm
def build_model(model_config, task, n_class, device, args):
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

    if task == "cifar10":
        model = resnet.super_resnet18(True, args.use_scaler, args.width_ratio_list).get_subnet(model_config).to(device)



    # if model_config == 'ResNet18':
    #     model = resnet.SuperResnet(resnet.DynamicBasicBlock,
    #                                4,
    #                                [2, 2, 4, 2],
    #                                [64, 128, 256, 512],
    #                                dummy_input.size(),
    #                                n_class,
    #                                [0.0, 0.125, 0.25, 0.5]
    #     )
    # elif model_config == 'DenseNet121':
    #     model = densenet.DenseNet121(n_class)
    # elif model_config == 'BERT':
    #     model = AutoAdapterModel.from_pretrained('bert-base-uncased')
    #     model.add_adapter("mnli")
    #     model.add_classification_head("mnli", num_labels=n_class)
    #     model.active_adapters = "mnli"
    # else:
    #     raise ValueError('Wrong model name:', model_config)


    model.dummy_input = dummy_input

    return model



