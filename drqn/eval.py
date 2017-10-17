#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import threading
import random
import numpy as np
import time
import argparse
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import matplotlib.pyplot as plt

import gym
import tensorflow as tf
import tflearn
from tensorflow.contrib import rnn as _rnn

writer_summary = tf.summary.FileWriter
merge_all_summaries = tf.summary.merge_all
histogram_summary = tf.summary.histogram
scalar_summary = tf.summary.scalar

game = 'SpaceInvaders-v0'
# Learning threads
n_threads = 8

# =============================
#   Training Parameters
# =============================
# Max training steps
TMAX = 80000000
# Current training step
T = 0
# Consecutive screen frames when performing training
action_repeat = 4
# Async gradient update frequency of each learning thread
I_AsyncUpdate = 20
# Timestep to reset the target network
I_target = 40000
# Learning rate
learning_rate = 0.001
# Reward discount rate
gamma = 0.99
# Number of timesteps to anneal epsilon
anneal_epsilon_timesteps = 400000
# Size of the epoch (used for reward measurement over time)
epoch_size = 2000000
epoch = 0
epoch_high_score = 0
high_score = []

# =============================
#   Utils Parameters
# =============================
# Display or not gym evironment screens
show_training = False
# Directory for storing tensorboard summaries
summary_dir = '/tmp/tflearn_logs/'
summary_interval = 100
checkpoint_path = 'qlearning.tflearn.ckpt'
checkpoint_interval = 10000
# Number of episodes to run gym evaluation
num_eval_episodes = 100

# =============================
#   TFLearn Deep Q Network
# =============================
def collect_args():
    chain = None
    parser = argparse.ArgumentParser(description="Pacman DRAQN learning options")
    parser.add_argument('--restore', '-r', help="Path to saved model checkpoint", 
                        default="")
    parser.add_argument('--network', '-n', help="Model to run for pacman: DQN, DRQN, DRAQN",
                        default='A3CLSTM')
    parser.add_argument('--learning_rate', '-lr', help="default starting learning rate",
                        default=0.1)
    parser.add_argument('--epsilon', '-e', help="default starting epsilon (greedy aggression)",
                        default=0.99)
    parser.add_argument('--mode', '-m', help="select either training or validation mode",
                        default='train')
    parser.add_argument('--save_path', '-c', help="path to save model checkpoints at",
                        default='./model')
    parser.add_argument('--meta', '-g', help="path to meta graph file",
                        default='./model')

    args = parser.parse_args()
    return args


class ModelMachine(object):
   
    def __init__(self):
	
	self.attention_state = None

    def build_dqn(self, num_actions, action_repeat):
	"""
	Building a DRQN.
	"""
	inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])
	# Inputs shape: [batch, channel, height, width] need to be changed into
	# shape [batch, height, width, channel]
	input      = tf.transpose(inputs, [0, 2, 3, 1])
	conv1      = tflearn.conv_2d(input, 32, 8, strides=4, activation='relu')
	conv2      = tflearn.conv_2d(conv1, 64, 4, strides=2, activation='relu')
	#fc1_game   = tflearn.fully_connected(conv2, 512, activation='relu')
	#q_values   = tflearn.fully_connected(fc1_game, num_actions)

	fc1_hidden = tflearn.fully_connected(conv2, 256, activation='relu')
	lstm_in    = tf.reshape(fc1_hidden, [-1, 1, 256])
	lstm_1     = tflearn.lstm(lstm_in, 256, dropout=0.5)
	q_values   = tflearn.fully_connected(lstm_1, num_actions, activation='softmax')
	return inputs, q_values

    def _lstm(self, incoming, n_units, activation='tanh', inner_activation='sigmoid',
         dropout=None, bias=True, weights_init=None, forget_bias=1.0,
         return_seq=False, return_state=False, initial_state=None,
         dynamic=False, trainable=True, restore=True, reuse=False,
         scope=None, name="LSTM"):
    	
	cell = tflearn.BasicLSTMCell(n_units, activation=activation,
                         inner_activation=inner_activation,
                         forget_bias=forget_bias, bias=bias,
                         weights_init=weights_init, trainable=trainable,
                         restore=restore, reuse=reuse)

	x = tflearn.layers._rnn_template_ex(incoming, cell=cell, dropout=dropout,
                      return_seq=return_seq, return_state=return_state,
                      initial_state=initial_state, dynamic=dynamic,
                      scope=scope, name=name)

    	return x

    def build_a3c(self, num_actions, action_repeat):
	"""
	Build an Async Advantage Actor Critic Network
	"""
	inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84])
	# input shape same as the DRQN 
	input  = tf.transpose(inputs, [0, 2, 3, 1])
        #print (input.shape)
	net  = tflearn.conv_2d(input, 32, 8, strides=1, activation='relu', padding='valid')
        #print (net.shape)
	#net  = tflearn.conv_2d(net, 64, 4, strides=1, padding='valid')
	# smaller FC layer to account for more versitle LSTM cell structure
	"""net    = tflearn.fully_connected(net, 256, activation='relu')
	 reshape larger fc output to lstm shape (1, batch_size, fc_size)
	#net = tf.reshape(net, [-1, 1, 256])

	lstm = tf.contrib.rnn.BasicLSTMCell(256)
	att  = tf.contrib.rnn.AttentionCellWrapper(lstm, 128, state_is_tuple=True)
        print (dir(att))

	net = self._lstm(net, 256, dynamic=True)
        #net, states = tflearn.bidirectional_rnn(net, l, l, dynamic=True, return_states=True)

        #net, states  = tf.nn.bidirectional_dynamic_rnn(cell_fw=att,
        #							   cell_bw=att,
        #							   dtype=tf.float32,
        #							   sequence_length=[1],
        #							   inputs=net)

        #self.attention_state = states
        

	# reshape lstm back to (batch_size, 256)
	net = tf.reshape(net, [-1, 256])
	q_values = tflearn.fully_connected(net, num_actions, activation='softmax')
        
	return inputs, q_values
        """ 
        return inputs, net
# =============================
#   ATARI Environment Wrapper
# =============================
class AtariEnvironment(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size action_repeat from which environment state is constructed.
    """
    def __init__(self, gym_env, action_repeat):
        self.env = gym_env
        self.action_repeat = action_repeat

        # Agent available actions, such as LEFT, RIGHT, NOOP, etc...
        self.gym_actions = range(gym_env.action_space.n)
        # Screen buffer of size action_repeat to be able to build
        # state arrays of size [1, action_repeat, 84, 84]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer.
        """
        # Clear the state buffer
        self.state_buffer = deque()

        frame = self.env.reset()
        frame = self.get_preprocessed_frame(frame)
        framestack = np.stack([frame for i in range(self.action_repeat)], axis=0)

        for i in range(self.action_repeat-1):
            self.state_buffer.append(frame)
        return framestack

    def get_preprocessed_frame(self, observation):
        """
        0) Atari frames: 210 x 160
        1) Get image grayscale
        2) Rescale image 110 x 84
        3) Crop center 84 x 84 (you can crop top/bottom according to the game)
        """
        return resize(rgb2gray(observation), (110, 84))[13:110 - 13, :]

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of action_repeat-1 previous
        frames and current one). Pops oldest frame, adds current frame to
        the state buffer. Returns current state.
        """

        obv, reward, done, info = self.env.step(self.gym_actions[action_index])
        obv = self.get_preprocessed_frame(obv)

        previous_frames = np.array(self.state_buffer)
        state_stack = np.empty((self.action_repeat, 84, 84))
        state_stack[:self.action_repeat-1, :] = previous_frames
        state_stack[self.action_repeat-1] = obv

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(obv)

        return state_stack, reward, done, info


def build_graph(num_actions, args):
    # Create shared deep q network
    M = ModelMachine()
    if str(args.network) == "A3CLSTM":
        print (num_actions)
        s, q_network = M.build_a3c(num_actions, action_repeat)
        #s = M.build_a3c(num_actions, action_repeat)

    q_values = q_network
    network_params = tf.trainable_variables()
    # Op for periodically updating target network with online network weights
    graph_ops = {"s": s, "q_values": q_values}
    #graph_ops = {"s": s}
    return graph_ops


# Set up some episode summary ops to visualize on tensorboard.
def build_summaries():
    episode_reward = tf.Variable(0.)
    scalar_summary("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    scalar_summary("Qmax Value", episode_ave_max_q)
    logged_epsilon = tf.Variable(0.)
    scalar_summary("Epsilon", logged_epsilon)
    high_score = tf.Variable(0.)
    scalar_summary("High Score", high_score)
    # Threads shouldn't modify the main graph, so we use placeholders
    # to assign the value of every summary (instead of using assign method
    # in every thread, that would keep creating new ops in the graph)
    summary_vars = [episode_reward, episode_ave_max_q, logged_epsilon, high_score]
    summary_placeholders = [tf.placeholder("float")
                            for i in range(len(summary_vars))]
    assign_ops = [summary_vars[i].assign(summary_placeholders[i])
                  for i in range(len(summary_vars))]
    summary_op = merge_all_summaries()
    return summary_placeholders, assign_ops, summary_op


def get_num_actions():
    """
    Returns the number of possible actions for the given atari game
    """
    # Figure out number of actions from gym env
    env = gym.make(game)
    num_actions = env.action_space.n
    return num_actions

def evaluation(session, graph_ops, saver, args):
    """
    Evaluate a model.
    """
    print ("attempting to restore from", args.restore)
    saver = tf.train.import_meta_graph(args.meta)
    saver.restore(session, args.restore)
    print("Restored model weights from ", args.restore)
    monitor_env = gym.make(game)
    from gym import wrappers
    monitor_env = wrappers.Monitor(monitor_env, "model/eval",force=True )
    #monitor_env.monitor.start("model/eval")
    # Unpack graph ops
    s = graph_ops["s"]
    q_values = graph_ops["q_values"]

    # Wrap env with AtariEnvironment helper class
    env = AtariEnvironment(gym_env=monitor_env,
                           action_repeat=action_repeat)

    for i_episode in xrange(num_eval_episodes):
        s_t = env.get_initial_state()
        ep_reward = 0
        terminal = False
        while not terminal:
            monitor_env.render()
            readout_t = q_values.eval(session=session, feed_dict={s : [s_t]})
            #readout_t = s.eval(session=session, feed_dict={s : [s_t]})
            print (s_t.shape)
            #print (readout_t)
            from PIL import Image
            im = np.reshape(s_t, (4, 84, 84, 1))
            im = np.reshape(im[0], (84, 84))
            i  = Image.fromarray(im)
            print (im)
            i.show()

            import sys
            sys.exit(0)
            action_index = np.argmax(readout_t)
            s_t1, r_t, terminal, info = env.step(action_index)
            s_t = s_t1
            ep_reward += r_t
        print(ep_reward)
    monitor_env.monitor.close()


def main(_):
    with tf.Session() as session:
        print ("collecting args")
        args = collect_args()
        num_actions = get_num_actions()
        print ("building graph")
        graph_ops = build_graph(num_actions, args)
        saver = tf.train.Saver(max_to_keep=5)
        if args.mode == "test":
            evaluation(session, graph_ops, saver, args)
        elif args.mode == "train":
            print ("training")
            train(session, graph_ops, num_actions, saver, args)

if __name__ == "__main__":
    tf.app.run()
