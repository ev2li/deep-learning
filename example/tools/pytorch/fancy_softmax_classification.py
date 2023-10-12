# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# For reproducibility
torch.manual_seed(1)

z = torch.rand(3,5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
# print(hypothesis)
y = torch.randint(5,(3,)).long()
y_one_hot = torch.zeros_like(hypothesis)
# print(y_one_hot)
y_one_hot.scatter(1, y.unsqueeze(1), 1)
# print(y_one_hot)
print(torch.log(F.softmax(z, dim=1)))

import torch

input = torch.randn(2,4)
print(input)
output = torch.zeros(2,5)
print(output)
index = torch.tensor([[3,1,2,0],[1,2,0,3]])
output = output.scatter(1, index, input)

print(output)

# 一般scatter用于生成onehot向量，如下所示
index = torch.tensor([[1],[2],[0],[3]])
onehot = torch.zeros(4,4)
onehot.scatter_(1, index, 1)
print(onehot)


src = torch.Tensor([[1,2,3,4],[5,6,7,8]])
idx1 = torch.LongTensor([[1,1,1,0],[2,2,2,1]])
torch.zeros([3,4]).scatter_(dim=0, index=idx1, src=src)  # 按行
