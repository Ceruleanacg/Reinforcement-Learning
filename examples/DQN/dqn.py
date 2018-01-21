# coding=utf-8

import tensorflow as tf
import numpy as np


np.random.seed(1)
tf.set_random_seed(1)


class DQN(object):
    def __init__(self, action_dim, state_dim, **options):

        self.action_dim = action_dim
        self.state_dim = state_dim

        try:
            self.alpha = options['alpha']
        except KeyError:
            self.alpha = 0.01

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.epsilon = options['epsilon']
        except KeyError:
            self.epsilon = 0.9

        try:
            self.reset_steps = options['reset_steps']
        except KeyError:
            self.reset_steps = 300

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 500

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        self.total_steps = 0

        self.buffer = np.zeros((self.buffer_size, state_dim + 1 + 1 + state_dim))

    def _init_nn(self):

        # Input state, state_next, reward, action.
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='input_state')
        self.state_next = tf.placeholder(tf.float32, [None, self.state_dim], name='input_state_next')
        self.reward = tf.placeholder(tf.float32, [None, 1], name='input_reward')
        self.action = tf.placeholder(tf.float32, [None, 1], name='input_action')

        # w,b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)
        b_initializer = tf.constant_initializer(0.1)

        # Evaluate net.
        with tf.variable_scope('evaluate_q_net'):
            phi_state = tf.layers.dense(self.state, 20, tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='phi_state_fc')
            self.q_value_predict = tf.layers.dense(phi_state, self.action_dim,
                                                   kernel_initializer=w_initializer,
                                                   bias_initializer=b_initializer,
                                                   name='Q_eval')

        with tf.variable_scope('target_q_net'):
            phi_state_next = tf.layers.dense(self.state_next, 20, tf.nn.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name='phi_state_next_fc')
            self.q_value_real = tf.layers.dense(phi_state_next, self.action_dim,
                                                kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer,
                                                name='Q_real')

        


