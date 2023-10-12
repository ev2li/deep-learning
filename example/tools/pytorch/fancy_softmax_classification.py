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