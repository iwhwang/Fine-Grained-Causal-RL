import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_layer(w, b):
    nn.init.kaiming_uniform_(w, nonlinearity='relu')
    fan_in = w.shape[1]
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(b, -bound, bound)


def forward_network(input, weights, biases, activation=F.relu):
    """
    given an input and a multi-layer networks (i.e., a list of weights and a list of biases),
        apply the network to each input, and return output
    the same activation function is applied to all layers except for the last layer
    """
    x = input
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = torch.bmm(x, w) + b
        if (i < len(weights) - 1) and activation:
            x = activation(x)
    return x


def forward_network_batch(inputs, weights, biases, activation=F.relu):
    """
    given a list of inputs and a list of ONE-LAYER networks (i.e., a list of weights and a list of biases),
        apply each network to each input, and return a list
    """
    x = []
    for x_i, w, b in zip(inputs, weights, biases):
        x_i = torch.bmm(x_i, w) + b
        if activation:
            x_i = activation(x_i)
        x.append(x_i)
    return x