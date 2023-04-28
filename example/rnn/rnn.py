# -*- coding: utf-8 -*-

import numpy as np
from cnn import element_wise_op
from activators import ReluActivator, IdentityActivator
from functools import reduce

class RecurrentLayer(object):
    def __init__(self, input_width, state_width,
                 activator, learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros(
            (state_width, 1)))  # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4,
            (state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4,
            (state_width, state_width))  # 初始化W


    def forward(self, input_array):
        """
        根据式『式2』进行前向计算
        :param input_array:
        :return:
        """
        self.times += 1
        state = (np.dot(self.U, input_array) +
                 np.dot(self.W, self.state_list[-1]))
        element_wise_op(state, self.activator.forward)
        self.state_list.append(state)


    def backward(self, sensitivity_array, activator):
        """
        实现BPTT算法
        :param sensitivity_array:
        :param activator:
        :return:
        """
        self.calc_delta(sensitivity_array, activator)
        self.calc_gradient()


    def update(self):
        """

        :return:
        """

    def calc_delta(self, sensitivity_array, activator):
        """

        :param sensitivity_array:
        :param activator:
        :return:
        """

    def calc_gradient(self):
        """

        :return:
        """
        pass