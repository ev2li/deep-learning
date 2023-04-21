#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
- Network神经网络对象，提供API接口，它由若干层对象组成以及连接对象组成
- Layer层对象，由多个节点组成
- Node节点对象计算和记录节点自身的信息(比如输出值𝑎、误差项等𝛿)，以及与这个节点相关的上下游的连接
- Connection每个连接对象都要记录该连接的权重
- Connectons仅仅作为Connnection的集合对象，提供一些集合操作
"""

import random
from functools import reduce
from numpy import exp

def sigmoid(inX):
    """
    激活函数
    :param inX:
    :return:
    """
    return 1.0 / (1 + exp(-inX))

class Node(object):
    """
    节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项(𝛿)的计算
    """
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        :param layer_index: 节点所属层的编号
        :param node_index: 节点编号
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0  # 输出
        self.delta = 0  # 误差项


    def set_output(self, output):
        """
        设置节点的输出值，如果节点属于输入层会用到这个函数
        :param output:
        :return:
        """
        self.output = output


    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        :param conn:
        :return:
        """
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        """
        添加一个到上游节点的连接
        :param conn:
        :return:
        """
        self.upstream.append(conn)


    def calc_output(self):
        """
        根据式1计算出节点的输出
        每个节点的输出算法，N元一次方程求和
        :return:
        """
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        #结果放入激活函数
        self.output = sigmoid(output)


    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层时，根据式4计算delta
        :return:
        """
        downstream_delta = reduce(
            lambda ret,conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta


    def calc_output_layer_delta(self, label):
        """
        节点属于输出层时，根据式3计算delta
        :param label:
        :return:
        """
        self.delta = self.output * (1 - self.output) * (label - self.output)


    def __str__(self):
        """
        打印节点信息
        :return:
        """
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str



class ConstNode(object):
    def __init__(self, layer_index, node_index):
        """
        构造节点对象
        layer_index: 节点所属层的编号
        node_index: 节点的编号
        :param layer_index:
        :param node_index:
        """
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0.0


    def append_downstream_connection(self, conn):
        """
        添加一个到下游节点的连接
        :param conn:
        :return:
        """
        self.downstream.append(conn)


    def calc_hidden_layer_delta(self):
        """
        节点属于隐藏层时，根据式4计算delta
        :return:
        """
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output *(1 - self.output) * downstream_delta


    def __str__(self):
        """
        打印节点信息
        :return:
        """
        node_str = '%u-%u: output: 1' %(self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        """
        初始化一层
        :param layer_index: 层的编号
        :param node_count: 层所包含的节点个数
        """
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))


    def set_output(self, data):
        """
        设置层的输出，当层是输入层时会用到
        :param data:
        :return:
        """
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])


    def calc_output(self):
        """
        计算层的输出向量
        :return:
        """
        for node in self.nodes[:-1]:
            node.calc_output()


    def dump(self):
        """
        打印层的信息
        :return:
        """
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        """
        初始化连接，权重初始化为一个很少的随机数
        :param upstream_node: 连接的上游节点
        :param downstream_node: 连接的下游节点
        """
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0


    def clac_gradient(self):
        """
        计算梯度
        :return:
        """
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        """
        获取当前的梯度
        :return:
        """
        return self.gradient


    def update_weight(self, rate):
        """
        根据梯度下降算法更新权重
        :param rate:
        :return:
        """
        self.clac_gradient()
        self.weight += rate * self.gradient


    def __str__(self):
        """
        打印连接信息
        :return:
        """
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []


    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for connection in self.connections:
            print(connection)


class Network(object):
    def __init__(self, layers):
        """
        初始化一个全连接神经网络
        :param layers: 二维数组，描述神经网络每层节点数
        """
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer + 1].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]

            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, iteration):
        """
        训练神经网络
        :param labels: 数组，训练样本标签，每个元素是一个样本标签
        :param data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        :param rate:
        :param iteration:
        :return:
        """
        for i in range(iteration):
            for d in range(len(data_set)):
                self.__train_one_sample(labels[d], data_set[d], rate)


    def __train_one_sample(self, label, sample, rate):
        """
        内部函数，用一个样本训练网络
        :param label:
        :param sample:
        :param rate:
        :return:
        """
        self.predict(sample)
        self.__calc_delta(label)
        self.__update_weight(rate)



    def predict(self, sample):
        """
        根据输入的样本预测输出值
        :param sample: 数组，样本的特征，也就是网络的输入向量
        :return:
        """
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))


    def __calc_delta(self, label):
        """
        内部函数，计算每个节的delta
        :param label:
        :return:
        """
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def __update_weight(self,rate):
        """
        内部函数，更新每个连接权重
        :param label:
        :return:
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)


    def __calc_gradient(self):
        """
        内部函数，计算每个连接的梯度
        :return:
        """
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()


    def get_gradient(self, label, sample):
        """
        获得网络在一个样本下，每个连接上的梯度
        :param label: 样本标签
        :param sample: 样本输入
        :return:
        """
        self.predict(sample)
        self.__calc_delta(label)
        self.__calc_gradient()


    def dump(self):
        """
        打印神经网络信息
        :return:
        """
        for layer in self.layers:
            layer.dump()


def gradient_check(network, sample_feature, sample_label):
    """
    梯度检查
    :param network: 神经网络对象
    :param sample_feature: 样本特征
    :param sample_label:  样本标签
    :return:
    """
    # 计算神经网络误差
    network_error = lambda vec1, vec2:  \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v:(v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)
    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


