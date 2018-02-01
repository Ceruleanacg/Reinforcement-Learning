# coding=utf-8

import tensorflow as tf
import numpy as np
import time
import gym


class DDPG(object):
    def __init__(self, action_space, state_space, action_upper_bound, **options):

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.tau = options['tau']
        except KeyError:
            self.tau = 0.01

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 10000

        try:
            self.session = options['session']
        except KeyError:
            self.session = tf.Session()

        self.action_space, self.state_space, self.action_upper_bound = action_space, state_space, action_upper_bound

        self.buffer = np.zeros((self.buffer_size, state_space * 2 + action_space + 1))
        self.buffer_item_count = 0

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_space], 'state')
        self.state_next = tf.placeholder(tf.float32, [None, self.state_space], 'state_next')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'reward')

    def _init_nn(self):
        with tf.variable_scope('predict'):
            self.a_predict = self.__build_actor_nn(self.state, True)
            self.q_predict = self.__build_critic(self.state, self.a_predict, True)
        with tf.variable_scope('next'):
            self.a_next = self.__build_actor_nn(self.state_next, False)
            self.q_next = self.__build_critic(self.state_next, self.a_next, False)

    def _init_op(self):

        self.a_p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict/actor')
        self.c_p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict/critic')

        self.a_n_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='next/actor')
        self.c_n_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='next/critic')

        self.actor_loss = -tf.reduce_mean(self.q_predict)
        self.actor_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.actor_loss,
                                                                                  var_list=self.a_p_params)

        self.q_target = self.reward + self.gamma * self.q_next
        self.q_loss = tf.losses.mean_squared_error(self.q_target, self.q_predict)
        self.critic_train_op = tf.train.AdamOptimizer(self.learning_rate * 2).minimize(self.q_loss,
                                                                                       var_list=self.c_p_params)

        self.update_q_net = [tf.assign(n, self.tau * p + (1 - self.tau) * n) for p, n in zip(self.c_p_params,
                                                                                             self.c_n_params)]
        self.update_a_net = [tf.assign(n, self.tau * p + (1 - self.tau) * n) for p, n in zip(self.a_p_params,
                                                                                             self.a_n_params)]

        self.session.run(tf.global_variables_initializer())

    def train(self):
        self.session.run([self.update_q_net, self.update_a_net])
        state, action, reward, state_next = self.get_sample_batch()
        self.session.run(self.actor_train_op, {self.state: state})
        self.session.run(self.critic_train_op, {
            self.state: state, self.a_predict: action, self.reward: reward, self.state_next: state_next
        })

    def get_next_action(self, state):
        action = self.session.run(self.a_predict, {self.state: state[np.newaxis, :]})
        return action[0]

    def get_sample_batch(self):
        indices = np.random.choice(self.buffer_size, size=self.batch_size)
        batch = self.buffer[indices, :]
        state = batch[:, :self.state_space]
        action = batch[:, self.state_space: self.state_space + self.action_space]
        reward = batch[:, -self.state_space - 1: -self.state_space]
        state_next = batch[:, -self.state_space:]
        return state, action, reward, state_next

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, action, [reward], state_next))
        index = self.buffer_item_count % self.buffer_size
        self.buffer[index, :] = transition
        self.buffer_item_count += 1

    def __build_actor_nn(self, state, trainable):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(.1)

        with tf.variable_scope('actor'):

            phi_state = tf.layers.dense(state,
                                        30,
                                        tf.nn.relu,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        trainable=trainable)

            action_prob = tf.layers.dense(phi_state,
                                          self.action_space,
                                          tf.nn.tanh,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init,
                                          trainable=trainable)

            # But why?
            return tf.multiply(action_prob, self.action_upper_bound)

    @staticmethod
    def __build_critic(state, action, trainable):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(.1)

        with tf.variable_scope('critic'):

            phi_state = tf.layers.dense(state,
                                        30,
                                        tf.nn.relu,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        trainable=trainable)

            phi_action = tf.layers.dense(action,
                                         30,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init,
                                         trainable=trainable)

            q_value = tf.layers.dense((phi_state + phi_action),
                                      1,
                                      kernel_initializer=w_init,
                                      bias_initializer=b_init,
                                      trainable=trainable)

            return q_value


def main(_):

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    action_upper_bound = env.action_space.high

    ddpg = DDPG(action_space, state_space, action_upper_bound)

    exploration_scale = 3

    for episode in range(200):

        state, reward_episode = env.reset(), 0

        for step in range(200):

            if reward_episode > 300:
                env.render()

            action = ddpg.get_next_action(state)
            action = np.clip(np.random.normal(action, exploration_scale), -2, 2)

            state_next, reward, done, info = env.step(action)

            ddpg.save_transition(state, action, reward / 10, state_next)

            if ddpg.buffer_item_count > ddpg.buffer_size:
                exploration_scale *= .9995
                ddpg.train()

            state = state_next

            reward_episode += reward

        print("Episode: {} | Reward: {} | Scale: {}".format(episode, reward_episode, exploration_scale))


if __name__ == '__main__':
    tf.app.run()
