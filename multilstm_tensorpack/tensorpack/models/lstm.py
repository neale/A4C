#!/usr/bin/env python

import tensorflow as tf
from .common import layer_register

__all__ = ['Lstm']

@layer_register()
def Lstm(x, num_inputs):


    lstm_input = tf.expand_dims(x, [0])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_inputs, state_is_tuple=True)

    c_init = tf.zeros([1, lstm_cell.state_size.c], tf.float32)  
    h_init = tf.zeros([1, lstm_cell.state_size.h], tf.float32) 

    state_init = [c_init, h_init]    
    state_in = tf.contrib.rnn.LSTMStateTuple(c_init, h_init) 
    lstm_init_shape = [c_init, h_init]       
    state_in = tf.contrib.rnn.LSTMStateTuple(c_init, h_init)  
    lstm_init_shape = [c_init, h_init] 

    lstm_out, lstm_out_state = tf.nn.dynamic_rnn(lstm_cell, lstm_input, sequence_length, 
                                                state_in, 
                                                dtype=tf.float32)         

    return (lstm_out, lstm_out_state, name='output')

