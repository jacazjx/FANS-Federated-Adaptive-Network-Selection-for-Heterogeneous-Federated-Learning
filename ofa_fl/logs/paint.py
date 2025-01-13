import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from hypernet.utils.evaluation import visualize_model_parameters


model_weight = torch.load("/mnt/sdb1/zjx/RecipFL/ofa_fl/logs/cifar10/seed4321/hypernet.pth")

model = {}
for k, v in model_weight.items():
    if "hidden_layer" in k:
        model[k] = v
# visualize_model_parameters(model)




# log_file_path = "/mnt/sdb1/zjx/RecipFL/ofa_fl/logs/cifar10/seed4321/alpha0.5.distill_depthfl.num_client_6.sample_client_6.2024-11-04_07:25:46.log"
log_file_path = "/mnt/sdb1/zjx/RecipFL/ofa_fl/logs/cifar10/seed4321/alpha0.5.fans.num_client_6.sample_client_6.2024-11-27_07:01:22.log"
'''
2024-11-04 07:36:28,092:CRITICAL: [TRAIN] Round 6, time=1.493mins, ACC-small=0.5338, ACC-large=0.3747
2024-11-04 07:36:28,092:CRITICAL: {0: {'ACC': 0.4748}}
2024-11-04 07:36:28,092:CRITICAL: {1: {'ACC': 0.6007}}
2024-11-04 07:36:28,092:CRITICAL: {2: {'ACC': 0.5764}}
2024-11-04 07:36:28,093:CRITICAL: {3: {'ACC': 0.5524}}
2024-11-04 07:36:28,093:CRITICAL: {4: {'ACC': 0.4646}}
2024-11-04 07:36:28,093:CRITICAL: {5: {'ACC': 0.3747}}
'''

with open(log_file_path, 'r') as file:
    lines = file.readlines()
num_round = 0
client_acc_data = {}  # 字典存储每个客户端的 ACC 值
for line in lines:
    client_data = re.findall(r"{(\d+): {'ACC': ([0-9.]+)}", line)
    for client_id, acc in client_data:
        client_id = int(client_id)  # 转换为整数
        acc = float(acc)  # 转换为浮点数
        if client_id not in client_acc_data:
            client_acc_data[client_id] = []  # 初始化客户端列表
        client_acc_data[client_id].append(acc)  # 添加 ACC 值
        num_round = len(client_acc_data[client_id])

# 打印每个客户端的 ACC 数据
for client_id, acc_values in client_acc_data.items():
    print(f'Client {client_id}: {acc_values}')

# 可选：计算均值、最小值和最大值并绘图
mean_acc = []
min_acc = []
max_acc = []

for idx in range(num_round):
    acc_values = [client_acc_data[client_id][idx] for client_id in client_acc_data.keys()]
    mean_acc.append(np.mean(acc_values))
    min_acc.append(np.min(acc_values))
    max_acc.append(np.max(acc_values))

client_acc_data['Mean'] = mean_acc
client_acc_data['Min'] = min_acc
client_acc_data['Max'] = max_acc

# 创建 DataFrame 并保存到 CSV
df = pd.DataFrame(client_acc_data)

# 保存到 CSV 文件
df.to_csv('result.csv')


# 绘制带范围的折线图
plt.figure(figsize=(10, 6), dpi=300)
x = range(num_round)

# 绘制每个客户端的均值
plt.plot(x, mean_acc, label='DepthFL', color='blue')

# 添加范围的阴影
plt.fill_between(x, min_acc, max_acc, color='blue', alpha=0.2)

# 添加图表细节
plt.title('Client ACC over Rounds')
plt.xlabel('Communication Round')
plt.ylabel('ACC')
# plt.xticks(x, np.arange(1, num_round, 5))  # round
plt.xlim(0, num_round)
plt.ylim(0, 1)
plt.legend()
plt.grid(linestyle='--')

plt.savefig('my_plot.pdf', dpi=300)
# 显示图表
plt.show()
