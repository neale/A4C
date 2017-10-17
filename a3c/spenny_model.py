
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class SpennyFC(torch.nn.Module):

    def __init__(self, num_inputs, action_space):

        num_outputs = 4

        self.A_fc1 = nn.Linear(256, 256)
        self.A_fc2 = nn.Linear(256, 256)

        self.c_linear = nn.Linear(256, 1)
        self.a_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.a_linear.weight.data = normalized_columns_initializer(
            self.a_linear.weight.data, 0.01)
        self.a_linear.bias.data.fill_(0)
        self.c_linear.weight.data = normalized_columns_initializer(
            self.c_linear.weight.data, 1.0)
        self.c_linear.bias.data.fill_(0)

        self.train()

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, batch, 256).zero_()),
                Variable(weight.new(1, batch, 256).zero_()))


    def forward(self, inputs):
        inputs, (h_in, c_in) = inputs
        x = F.elu(self.A_fc1(inputs))
        x = F.elu(self.A_fc2(x))
        y = F.elu(self.c_linear(x))
        x = F.elu(self.a_linear(x))


        # flatten somewhere here

        return x, y
