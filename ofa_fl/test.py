import torch
import numpy as np
# from hypernet.utils.evaluation import visualize_model_parameters
# # model_state_dict = torch.load("/mnt/sdb1/zjx/hypernet.pth")
# model_state_dict = torch.load("/mnt/sdb1/zjx/RecipFL/ofa_fl/logs/cifar10/seed4321/hypernet.pth")
# visualize_model_parameters(model_state_dict)



# x = torch.randn(64, 128, 3, 3)
# p = 0.5
#
# mask = torch.bernoulli(torch.full_like(x[1, :, 1, 1], p))
# print(mask)

block_flag = torch.tensor([0.25, 1, 0.75, 1, 1, 0.5, 1, 1])
block_width = torch.tensor([64, 64, 128, 128, 256, 256, 512, 512])

block_prop = block_flag * block_width / block_width.sum()
print(block_prop.sum())

