# -*- coding: utf-8 -*-


def MMOE(dnn_feature_columns, num_experts=3, expert_dnn_hidden_units=(256, 128), tower_dnn_hidden_units=(64,),
         gate_dnn_hidden_units=(), l2_reg_embedding=0.00001, l2_reg_dnn=0, dnn_dropout=0, dnn_activation='relu',
         dnn_use_bn=False, task_types=('binary', 'binary'), task_names=('ctr', 'ctcvr')):
    num_tasks = len(task_names)

    # 构建Input层并将Input层转成列表作为模型的输入
    input_layer_dict = build_input_layers(dnn_feature_columns)
    input_layers = list(input_layer_dict.values())

    # 筛选出特征中的sparse和Dense特征， 后面要单独处理
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns))

    # 获取Dense Input
    dnn_dense_input = []
    for fc in dense_feature_columns:
        dnn_dense_input.append(input_layer_dict[fc.name])

    # 构建embedding字典
    embedding_layer_dict = build_embedding_layers(dnn_feature_columns)
    # 离散的这些特特征embedding之后，然后拼接，然后直接作为全连接层Dense的输入，所以需要进行Flatten
    dnn_sparse_embed_input = concat_embedding_list(sparse_feature_columns, input_layer_dict, embedding_layer_dict,
                                                   flatten=False)

    # 把连续特征和离散特征合并起来
    dnn_input = combined_dnn_input(dnn_sparse_embed_input, dnn_dense_input)

    # 建立专家层
    expert_outputs = []
    for i in range(num_experts):
        expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022,
                             name='expert_' + str(i))(dnn_input)
        expert_outputs.append(expert_network)

    expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(expert_outputs)

    # 建立多门控机制层
    mmoe_outputs = []
    for i in range(num_tasks):  # num_tasks=num_gates
        # 建立门控层
        gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022,
                         name='gate_' + task_names[i])(dnn_input)
        gate_out = Dense(num_experts, use_bias=False, activation='softmax', name='gate_softmax_' + task_names[i])(
            gate_input)
        gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

        # gate multiply the expert
        gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                 name='gate_mul_expert_' + task_names[i])([expert_concat, gate_out])

        mmoe_outputs.append(gate_mul_expert)

    # 每个任务独立的tower
    task_outputs = []
    for task_type, task_name, mmoe_out in zip(task_types, task_names, mmoe_outputs):
        # 建立tower
        tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=2022,
                           name='tower_' + task_name)(mmoe_out)
        logit = Dense(1, use_bias=False, activation=None)(tower_output)
        output = PredictionLayer(task_type, name=task_name)(logit)
        task_outputs.append(output)

    model = Model(inputs=input_layers, outputs=task_outputs)
    return model
