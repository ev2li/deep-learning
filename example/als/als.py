# -*- coding: UTF-8 -*-

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import math
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def __get_pairs(n_users, n_items, dataset):
    """
    Get the pairs of users and items.
    获取数据匹配
    :param n_users:
    :param n_items:
    :param dataset:
    :return:
    """
    ndataset = []
    for data in dataset:
        u, i, r, _ = data.strip().split(',')
        if u == 'userId':
            continue
        u, i, r = int(u), int(i), int(float(r))
        if u > n_users:
            n_users = u
        if i > n_items:
            n_items = i
        ndataset.append((u, i, r))

    return ndataset, n_users, n_items


def read_datas(input_path, file_name, test_ratio=0.1):
    """
    读取数据
    :param input_path:
    :param file_name:
    :param test_ratio:
    :return:
    """
    with open(input_path + file_name, 'r', encoding="utf-8") as f:
        all_datas = f.read().split("\n")
    all_datas = list(set(filter(None, all_datas)))
    test_datas = np.random.choice(all_datas, int(len(all_datas) * test_ratio), replace=False)
    train_datas = list(set(all_datas) - set(test_datas))
    n_users, n_items = 0, 0
    test_datas, n_users, n_items = __get_pairs(n_users, n_items, test_datas)
    train_datas, n_users, n_items = __get_pairs(n_users, n_items, train_datas)
    return train_datas, test_datas, n_users + 1, n_items + 1


class ALS(nn.Module):
    def __init__(self, n_users, n_items, dim=50):
        """
        ALS的pytorch版本
        :param n_users:
        :param n_items:
        :param dim:
        """
        super(ALS, self).__init__()
        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.items = nn.Embedding(n_items, dim, max_norm=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, i):
        u = self.users(u)
        i = self.items(i)
        ui = torch.sum(u * i, axis=1)
        logit = self.sigmoid(ui)
        return logit

def MSE(y_true, y_pred):
    return np.average((np.array(y_true) - np.array(y_pred)) ** 2)
def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5
def evalution(model, dataset):
    """
    模型评估
    :param model:
    :param dataset:
    :return:
    """
    label_sigmoid = nn.Sigmoid()
    u, i, r = dataset[:, 0], dataset[:, 1], dataset[:, 2]
    with torch.no_grad():
        output = model(u, i)
    y_pred = np.array(output.cpu().detach().numpy())
    y_true = np.array(label_sigmoid(r).cpu().detach().numpy())
    rmse = RMSE(y_true, y_pred)
    return rmse
def train(input_path, file_name, epochs=20, batch_size=1024):
    """
    lfm推荐算法训练
    :param input_path:
    :param file_name:
    :param epochs: 迭代次数
    :param batch_size:批量数据大小
    :return:
    """
    print("读取数据...")
    train_data, test_data, n_users, n_items = read_datas(input_path, file_name)
    # 初始化als模型
    print("初始化als模型...")
    model = ALS(n_users, n_items)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    criterion = torch.nn.BCELoss()
    model.cuda()
    train_data = torch.LongTensor(train_data).cuda()
    test_data = torch.LongTensor(test_data).cuda()
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    label_sigmoid = nn.Sigmoid()
    print("开始训练...")
    for e in range(epochs):
        all_loss = 0
        for dataset in iter(dataloader):
            u, i, r = dataset[:, 0], dataset[:, 1], dataset[:, 2]
            optimizer.zero_grad()
            r = r.float()
            result = model(u, i)
            r = label_sigmoid(r)
            loss = criterion(result, r)
            all_loss += loss
            loss.backward()
            optimizer.step()
        print('epoch{}, avg_loss={}'.format(e, all_loss / (len(train_data) // batch_size)))
        rmse = evalution(model, test_data)
        print('RMSE:{}'.format(round(rmse, 3)))
    torch.save(model, "models/pytorch_lfm.model")
def reverse_sigmoid(x):
    return -math.log(1 / (x + 1e-8) - 1)
def sigmoid(x):
    """
    sigmoid参数
    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))
def predict_single(u_index, i_index, model_path):
    """
    预测单个数据
    :param u_index:
    :param i_index:
    :param model_path:
    :return:
    """
    u = torch.LongTensor([u_index]).to(device)
    i = torch.LongTensor([i_index]).to(device)
    model = torch.load(model_path)
    result = model(u, i)
    result = result.cpu().detach().numpy()[0]
    # result = reverse_sigmoid(result)
    print(result)
if __name__ == '__main__':
    input_path = "datas/ml-25m/"
    file_name = "ratings.csv"
    train(input_path, file_name)
    # predict_single(1, 296, "models/pytorch_lfm.model")