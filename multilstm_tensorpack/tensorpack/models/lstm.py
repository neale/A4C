#!/usr/bin/env python

import tensorflow as tf
from .common import layer_register


__all__ = ['Lstm']

@layer_register()
def Lstm(x, num_inputs, use_peepholes=False,
         cell_clip=None, initializer=None,
         forget_bias=1.0,  state_is_tuple=True,
               activation=tf.tanh):

    return (tf.contrib.rnn.LSTMCell(num_inputs, use_peepholes, cell_clip, forget_bias))

