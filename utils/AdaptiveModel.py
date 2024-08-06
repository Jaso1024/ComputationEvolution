import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm
import pickle
import random
import copy
import random
from collections import defaultdict
import concurrent.futures
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import hashlib
import torch.multiprocessing as mp


class AdaptiveComputationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AdaptiveComputationLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.operations = []
        self.params = nn.ParameterDict()
        self.nodes = []
        self.edges = defaultdict(list)
        self.nodes.append("input")
        self.add_matmul(self.add_weight_matrix((self.input_dim, self.output_dim)), 'input')

    def add_node(self, name):
        self.nodes.append(name)
        return name

    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)

    def add_weight_matrix(self, shape):
        param_name = (f'weight_{sum(1 for name in self.params if name.startswith("weight"))}')
        self.params[param_name] = nn.Parameter(torch.randn(shape)).to('cuda')
        return param_name

    def add_bias(self, shape):
        param_name = f'bias_{sum(1 for name in self.params if name.startswith("bias"))}'
        self.params[param_name] = nn.Parameter(torch.randn(shape)).to('cuda')
        return param_name

    def add_matmul(self, weight_matrix_name, input_name):
        output_name = self.add_node(f"matmul_{len(self.operations)}")
        def matmul_op(x, nodes):
            return torch.matmul(nodes[input_name], self.params[weight_matrix_name])

        
        matmul_op.name = f"{output_name} = {weight_matrix_name} * {input_name}"
        
        self.operations.append((matmul_op, output_name, input_name))
        self.add_edge(input_name, output_name)
        

    def add_addition(self, input_name1, input_name2):
        output_name = self.add_node(f"add_{len(self.operations)}")
        def addition_op(x, nodes):
            return nodes[input_name1] + nodes[input_name2]
        addition_op.name = f"{output_name} = {input_name1} + {input_name2}"
        self.operations.append((addition_op, output_name, (input_name1, input_name2)))
        self.add_edge(input_name1, output_name)
        self.add_edge(input_name2, output_name)

    def add_bias_addition(self, bias_name, input_name):
        output_name = self.add_node(f"bias_add_{len(self.operations)}")
        def bias_addition_op(x, nodes):
            return nodes[input_name] + self.params[bias_name]
        bias_addition_op.name = f"{output_name} = {input_name} + {bias_name}"
        self.operations.append((bias_addition_op, output_name, input_name))
        self.add_edge(input_name, output_name)

    def add_division(self, input_name, divisor):
        output_name = self.add_node(f"div_{len(self.operations)}")
        def division_op(x, nodes):
            return nodes[input_name] / divisor
        division_op.name = f"{output_name} = {input_name} / {divisor}"
        self.operations.append((division_op, output_name, input_name))
        self.add_edge(input_name, output_name)

    def add_activation(self, activation_func, func_name, input_name):
        output_name = self.add_node(f"{func_name}_{len(self.operations)}")
        def activation_op(x, nodes):
            return activation_func(nodes[input_name])
        activation_op.name = f"{output_name} = {func_name}({input_name})"
        self.operations.append((activation_op, output_name, input_name))
        self.add_edge(input_name, output_name)

    def add_custom_operation(self, operation, op_name, input_name):
        output_name = self.add_node(f"{op_name}_{len(self.operations)}")
        def custom_op(x, nodes):
            return operation(nodes[input_name])
        custom_op.name = f"{output_name} = {op_name}({input_name})"
        self.operations.append((custom_op, output_name, input_name))
        self.add_edge(input_name, output_name)

    def add_relu(self, input_name):
        self.add_activation(F.relu, "relu", input_name)

    def add_gelu(self, input_name):
        self.add_activation(F.gelu, "gelu", input_name)

    def add_leaky_relu(self, input_name, negative_slope=0.01):
        self.add_activation(lambda x: F.leaky_relu(x, negative_slope=negative_slope), "leaky_relu", input_name)

    def add_tanh(self, input_name):
        self.add_activation(torch.tanh, "tanh", input_name)

    def add_sigmoid(self, input_name):
        self.add_activation(torch.sigmoid, "sigmoid", input_name)

    def add_softmax(self, input_name, dim=None):
        self.add_activation(lambda x: F.softmax(x, dim=dim), "softmax", input_name)

    def add_softplus(self, input_name):
        self.add_activation(F.softplus, "softplus", input_name)

    def add_softsign(self, input_name):
        self.add_activation(F.softsign, "softsign", input_name)

    def add_elu(self, input_name, alpha=1.0):
        self.add_activation(lambda x: F.elu(x, alpha=alpha), "elu", input_name)

    def add_selu(self, input_name):
        self.add_activation(F.selu, "selu", input_name)

    def add_swish(self, input_name):
        self.add_activation(lambda x: x * torch.sigmoid(x), "swish", input_name)

    def add_hardshrink(self, input_name, lambd=0.5):
        self.add_activation(lambda x: F.hardshrink(x, lambd=lambd), "hardshrink", input_name)

    def add_hardtanh(self, input_name, min_val=-1.0, max_val=1.0):
        self.add_activation(lambda x: F.hardtanh(x, min_val=min_val, max_val=max_val), "hardtanh", input_name)

    def add_logsigmoid(self, input_name):
        self.add_activation(F.logsigmoid, "logsigmoid", input_name)

    def add_rrelu(self, input_name, lower=0.125, upper=0.333, training=False):
        self.add_activation(lambda x: F.rrelu(x, lower=lower, upper=upper, training=training), "rrelu", input_name)

    def mutate(self):
        mutation_type = random.choice([
            self._mutate_add_operation,
            self._mutate_modify_parameter,
            self._add_weight_matrix_mutation, 
            self._add_matmul_mutation,
            self._add_bias_mutation,
            self._add_addition_mutation,
        ])
        mutation_type()

    def _mutate_add_operation(self):
        operation_type = random.choice([
            self.add_relu, self.add_gelu, self.add_leaky_relu, self.add_tanh,
            self.add_sigmoid, self.add_softmax, self.add_softplus, self.add_softsign,
            self.add_elu, self.add_selu, self.add_swish, self.add_hardshrink,
            self.add_hardtanh, self.add_logsigmoid, self.add_rrelu, 
        ])
        input_name = random.choice(self.nodes)
        operation_type(input_name)

    def _mutate_remove_operation(self):
        if len(self.operations) > 2:
            index = random.randint(0, len(self.operations) - 1)
            operation = self.operations.pop(index)
            output_name = operation[1]
            self.edges = defaultdict(list, {k: [n for n in v if n != output_name] for k, v in self.edges.items()})

    def _mutate_modify_parameter(self):
        if self.params:
            param_name = random.choice(list(self.params.keys()))
            param = self.params[param_name]
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.1)

    def _add_weight_matrix_mutation(self):
        shape = (self.input_dim, self.output_dim)
        self.add_weight_matrix(shape)

    def _add_matmul_mutation(self):
        if sum(1 for name in self.params if name.startswith('weight')) > 0:
            weight_matrix_name = f'weight_{random.randint(0, sum(1 for name in self.params if name.startswith("weight")) - 1)}'
            input_name = random.choice(self.nodes)
            self.add_matmul(weight_matrix_name, input_name)

    def _add_bias_mutation(self):
        shape = (self.output_dim,)
        bias_name = self.add_bias(shape)
        input_name = random.choice(self.nodes)
        self.add_bias_addition(bias_name, input_name)

    def _add_addition_mutation(self):
        if len(self.nodes) > 1:
            input_name1, input_name2 = random.sample(self.nodes, 2)
            self.add_addition(input_name1, input_name2)

    def visualize_operations(self):
        visualization = ""
        for i, (op, output_name, input_name) in enumerate(self.operations):
            if isinstance(input_name, tuple):
                visualization += f"Line {i+1}: {op.name} ({input_name[0]}, {input_name[1]} ==> {output_name})\n"
            else:
                visualization += f"Line {i+1}: {op.name} ({input_name} ==> {output_name})\n"
        return visualization

    def forward(self, x):
        nodes = {'input': x}
        for op, output_name, input_name in self.operations:
            if isinstance(input_name, tuple):
                nodes[output_name] = op(x, nodes)
            else:
                nodes[output_name] = op(x, nodes)
        return nodes[self.operations[-1][1]]


class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.adaptive_layer = AdaptiveComputationLayer(64, 64)
        self.layer2 = nn.Linear(64, output_dim)
        self.flatten = nn.Flatten()
        self.device = 'cuda'

    def forward(self, x):
        x = self.layer1(self.flatten(x))
        x = self.adaptive_layer(x)
        x = self.layer2(x)
        return x
