#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import sys
import re
import time
import random
import uuid
import argparse
import multiprocessing
import threading
import cv2
from collections import deque
import six
from six.moves import queue

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import *
from simulator import *
import common
from common import (play_model, Evaluator, eval_model_multithread)

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 2
PREDICTOR_THREAD = None
EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = None
ENV_NAME = None


def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)

    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl


common.get_player = get_player

rnn = tf.contrib.rnn

class MySimulatorWorker(SimulatorProcess):

    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    class LSTMState(object):
        def __init__(self, name, initial_states, output_states, sequence_lengths, **kwargs):
            self._name = name
            self._initial_states = initial_states
            self._v_initial_states = [np.zeros(s.get_shape().as_list(), dtype=s.dtype.as_numpy_dtype()) for s in initial_states]
            self._output_states = output_states
            self._seq_len = sequence_lengths

            for k, v in kwargs.items():
                setattr(self, '_' + k, v)

        def reset_state(self, idx):
            for v in self._v_initial_states:
                v[idx] = np.zeros_like(v[idx])

        def update_state(self, indexs, newstates):
            for _idx in xrange(indexs.shape[0]):
                agent_index = indexs[_idx]
                for vidx, v in enumerate(self._v_initial_states):
                    self._v_initial_states[vidx][agent_index] = newstates[vidx][_idx]

    def __init__(self, **kwargs):

	from collections import OrderedDict
        self._lstm_states = OrderedDict()
        self.batch_size = BATCH_SIZE
        self.update_lstm_states = None
        self.reset_lstm_states = None
        self._input_agent_idxs = None
        self._input_states_list = []
        self._input_actions = None
        self._input_Rs = None
        self._input_tds = None
        self._input_is_over = None

        self._trainable_weights = None
        self._is_training = None

        self._kernel_update_lstm_states = None
        self._kernel_reset_lstm_states = None

    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward'),
		InputDesc(tf.int32, (None,), 'sequencelength')]
    
    def _get_dqn(self, image):
        image = image / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        
        return l 

    def _create_lstm_from_cell(self, cell, lstm_in, name='default_lstm', sequence_length=512):
        
	assert (name not in self._lstm_states)
        dtype = lstm_in.dtype
        tc = get_current_tower_context()
        states_name_prefix = 'train' if tc.is_training else 'predictor'
        
	if self.alstm_var_state is not None:
            states_name_prefix = self.lstm_state_var

        rnn_state_size = cell.state_size
        
	if isinstance(lstm_state_size, rnn.LSTMStateTuple):
            lstm_state_size = (lstm_state_size,)

        def get_state_variable(_name, state_size):
            
	    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                scope = tf.get_variable_scope()
                assert (scope.reuse == False)
                return tc.get_variable_on_tower(states_name_prefix + '/' + name + '/' + _name, 
		           shape=(self._batch_size, state_size), 
		           dtype=dtype, trainable=False, 
		           initializer=tf.zeros_initializer())

        init_lstm_states = []
        lstm_states = []
        states = []

        def get_lstm_state(state_size):
            
	    if isinstance(state_size, rnn.LSTMStateTuple):
                
		lstm_state_tuple_idx = len(array_lstm_state_tuple)
                c = get_state_variable('LSTMStateTuple-{}/c'.format(lstm_state_tuple_idx), state_size.c)
                h = get_state_variable('LSTMStateTuple-{}/h'.format(lstm_state_tuple_idx), state_size.h)
                
		initial_rnn_states.append(c)
                initial_rnn_states.append(h)
                ret = rnn.LSTMStateTuple(tf.gather(c, self._input_agent_idxs),
                                                     tf.gather(h, self._input_agent_idxs))
                lstm_states.append(ret)
                return ret

            elif isinstance(state_size, int):
                
		state_idx = len(array_state)
                s = get_state_variable('LSTMState-{}'.format(state_idx), state_size)
                init_lstm_states.append(s)
                ret = tf.gather(s, self._input_agent_idxs)
                states.append(ret)
                return ret

            elif isinstance(state_size, tuple):
                
		ret = []
                for idx, _state_size in enumerate(state_size):
                    ret.append(get_lstm_state(_state_size))
                ret = tuple(ret)
                return ret

            else: raise ValueError('unknown type {}'.format(type(state_size)))

        _init_lstm_states = get_lstm_state(cell.state_size)

        if seq_len is None: seq_len = self._input_seq_len
        lstm_outputs, lstm_out_state = tf.nn.dynamic_rnn(cell,
                                                           lstm_in,
                                                           initial_state=_init_lstm_states,
                                                           sequence_length=seq_len,
                                                           time_major=False,
                                                           scope=name,
                                                           )
        lstm_out_states = []
        
	def get_lstm_out_states(output):
        
	    if isinstance(output, rnn.LSTMStateTuple):
                lstm_out_states.append(output.c).append(output.h)

            elif isinstance(output, tuple):
                for idx, _output in enumerate(output):
                    get_lstm_out_states(_output)

            elif isinstance(output, tf.Tensor):
                lstm_out_states.append(output)

            else: raise ValueError('bad type {}'.format(type(output)))

        get_lstm_out_states(lstm_out_states)
        assert(len(lstm_out_states) == len(init_lstm_states))

        kernel_resets = []
        if tc.is_training:
            need_reset_states = tf.reshape(tf.ones_like(self._input_is_over) - self._input_is_over, (-1, 1))
            kernel_updates = [tf.scatter_update(init_lstm_states[idx], self._input_agent_idxs, 
			  lstm_out_states[idx] * tf.cast(need_reset_states, 
			  lstm_out_states[idx].dtype)) \
                          for idx in range(len(lstm_out_states))]

        else:
            # in predict mode, the is_over is for last state
            batch_size = tf.shape(self._input_agent_idxs)[0]
            kernel_updates = []
            for idx in range(len(init_lstm_states)):
                shape_states = tf.shape(init_lstm_states[idx])
                kernel = tf.scatter_update(initial_rnn_states[idx], self._input_agent_idxs, 
		      		       tf.zeros((batch_size,shape_states[1]), 
				       dtype=init_lstm_states[idx].dtype))
                kernel_resets.append(kernel)
                kernel = tf.scatter_update(init_lstm_states[idx], self._input_agent_idxs, lstm_out_states[idx])
                kernel_updates.append(kernel)
        
	self._lstm_states[name] = self.RNNStateInfo(name, init_lstm_states, lstm_out_states, 
						   seq_len, kernel_update_state=kernel_updates, 
						   kernel_reset_state=kernel_resets)

        return lstm_outputs   

    def _build_graph(self, inputs, batch_size=BATCH_SIZE, alstm_state_var=None):

        self._is_training       = is_training = get_current_tower_context() and get_current_tower_context().is_training
        self._model_inputs      = inputs
        self._input_agent_idxs  = inputs[0]
        self._input_actions     = inputs[1]
        self._input_Rs          = inputs[2]
        self._input_tds         = inputs[3]
        self._input_is_over     = inputs[4]
        self._input_seq_len     = inputs[5]
        self._input_states_list = inputs[6:]
        self._batch_size        = batch_size
        self._alstm_state_var   = alstm_state_var
        
        if len(self._lstm_states) > 0:
            kernel_update_states = []
            for s in self._lstm_states.values():
                kernel_update_states += s._kernel_update_state
            self._kernel_update_lstm_states = tf.group(*kernel_update_states, name='update_lstm_states')
            kernel_reset_states = []
            for s in self._lstm_states.values():
                kernel_reset_states += s._kernel_reset_state
            if len(kernel_reset_states) > 0:
                self._kernel_reset_lstm_states = tf.group(*kernel_reset_states, name='reset_lstm_states')

        state, action, futurereward = inputs
        model         = self._get_dqn(state)
        
        lstm_input    = tf.reshape(model, [1,-1,512])
        lstm_cell     = rnn.LSTMCell(512, use_peepholes=True)
	lstm_outputs  = _create_lstm_from_cell(lstm_cell, lstm_input)

        policy        = FullyConnected('fc-pi', lstm_outputs, out_dim=NUM_ACTIONS, nl=tf.identity)
        self.value    = FullyConnected('fc-v', lstm_outputs, 1, nl=tf.identity)
        
        self.value    = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.logits   = tf.nn.softmax(policy, name='logits')

        expf          = tf.get_variable('explore_factor', shape=[],
                               		initializer=tf.constant_initializer(1), trainable=False)
        logitsT       = tf.nn.softmax(policy * expf, name='logitsT')
        is_training   = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs     = tf.log(self.logits + 1e-6)

        log_pi_a_s    = tf.reduce_sum(log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage     = tf.subtract(tf.stop_gradient(self.value), futurereward, name='advantage')
        policy_loss   = tf.reduce_sum(log_pi_a_s * advantage, name='policy_loss')
        xentropy_loss = tf.reduce_sum(self.logits * log_probs, name='xentropy_loss')
        value_loss    = tf.nn.l2_loss(self.value - futurereward, name='value_loss')

        pred_reward   = tf.reduce_mean(self.value, name='predict_reward')
        advantage     = symbf.rms(advantage, name='rms_advantage')
        entropy_beta  = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost     = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost     = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')

        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage, self.cost)

	self.lstm_output = rnn.LSTMStateTuple(tf.zeros([1, 512]), tf.zeros([1, 512]))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)

    def _setup_graph(self):
        self.async_predictor = MultiThreadAsyncPredictor(
            self.trainer.get_predictors(['state'], ['logitsT', 'pred_value'],
                                        PREDICTOR_THREAD), batch_size=15)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        def cb(outputs):
            distrib, value = outputs.result()
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            client.memory.append(TransitionExperience(state, action, None, value=value))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []


def get_config():
    logger.auto_set_dir()
    M = Model()

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(80, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            ScheduledHyperParamSetter('explore_factor',
                                      [(80, 2), (100, 3), (120, 4), (140, 5)]),
            master,
            StartProcOrThread(master),
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits']), 2),
        ],
        session_config=get_default_sess_config(0.5),
        model=M,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--env', help='env', required=True)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    args = parser.parse_args()

    ENV_NAME = args.env
    assert ENV_NAME
    p = get_player()
    del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task != 'train':
        cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_names=['state'],
            output_names=['logits'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        nr_gpu = get_nr_gpu()
        if nr_gpu > 0:
            if nr_gpu > 1:
                predict_tower = range(nr_gpu)[-nr_gpu // 2:]
            else:
                predict_tower = [0]
            PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
            train_tower = range(nr_gpu)[:-nr_gpu // 2] or [0]
            logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
            trainer = AsyncMultiGPUTrainer
        else:
            logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
            nr_gpu = 0
            PREDICTOR_THREAD = 1
            predict_tower, train_tower = [0], [0]
            trainer = QueueInputTrainer
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()
