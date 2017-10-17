import math

import numpy as np

from scipy.misc import imshow
import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu, tanh, softmax
from torch.autograd import Variable
from attention_cell import attention_cell

# initialization scheme from https://discuss.pytorch.org/t/weight-initilzation/157/11
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

class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class A3C(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 1, stride=1, padding=1)
        # self.lstm = StackedLSTM(2, 32 * 3 * 3, 256, .5)
        self.lstm1 = nn.LSTMCell(256, 256)

        self.lstm = nn.LSTMCell(288, 256)
        self.dropout = nn.Dropout(.5)

        """ Attention Network"""
        #self.attn = attention_cell(256)
        self.A_fc1 = nn.Linear(256, 256)
        self.A_fc2 = nn.Linear(256, 256)

        num_outputs = action_space.n
        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor.weight.data = normalized_columns_initializer(self.actor.weight.data, 0.01)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

	self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def init_hidden(self, batch):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, batch, 256).zero_()),
                Variable(weight.new(1, batch, 256).zero_()))


    def forward(self, inputs):
        inputs, (h_in, c_in) = inputs
        x = leaky_relu(self.conv1(inputs))
        x = leaky_relu(self.conv2(x))
        x=  leaky_relu(self.conv3(x))
        x = leaky_relu(self.conv4(x))

        outputs = []

        import sys
        fmap = self.conv4.weight.view(-1, 1, 256)#, 32*3*3)

        for i, v_t in enumerate(fmap):
            v_t = v_t.view(1, 256)
            fc1_out = self.A_fc1(v_t)
            add = fc1_out + h_in
            tan = tanh(add)
            sm  = softmax(self.A_fc2(tan))
            #m   = sm.data.max()
            #sm.data = torch.div(sm.data, m)
            #sm.data = sm.data * v_t.data
            gv_t = torch.mul(sm, v_t)
            outputs.append(gv_t)

        g_t = torch.stack(outputs)
        g_t = g_t.sum(0).view(-1, 256)

        h_t, c_t = self.lstm1(g_t, (h_in, c_in))
        x = h_t

        return self.critic(x), self.actor(x), (h_t, c_t)
