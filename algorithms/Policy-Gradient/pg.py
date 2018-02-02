# coding=utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym

np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient(object):

    def __init__(self, action_dim, state_dim,  **options):

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.02

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.95

        try:
            self.session = options['session']
        except KeyError:
            self.session = tf.Session()

        self.action_dim, self.state_dim = action_dim, state_dim

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []

        self.train_steps = 0

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        with tf.variable_scope('input'):
            self.state = tf.placeholder(tf.float32,  [None, self.state_dim], name='state')
            self.action = tf.placeholder(tf.int32,   [None, ],               name='action')
            self.reward = tf.placeholder(tf.float32, [None, ],               name='reward')

    def _init_nn(self):

        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        with tf.variable_scope('nn'):

            phi_state = tf.layers.dense(self.state,
                                        10,
                                        tf.nn.tanh,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init)

            action_value_predict = tf.layers.dense(phi_state,
                                                   self.action_dim,
                                                   kernel_initializer=w_init,
                                                   bias_initializer=b_init)

            self.action_value_predict = action_value_predict
            self.action_prob = tf.nn.softmax(self.action_value_predict)

    def _init_op(self):

        with tf.variable_scope('loss'):
            # To maximize: R(θ) = Sum(R(τ) * P(τ|θ)).
            action_one_hot = tf.one_hot(self.action, self.action_dim)
            negative_cross_entropy = -tf.reduce_sum(tf.log(self.action_prob) * action_one_hot, axis=1)
            # negative_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.action_value_predict,
            #                                                                         labels=self.action)
            self.loss = tf.reduce_mean(negative_cross_entropy * self.reward)
            # self.loss = tf.reduce_mean(negative_cross_entropy)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.session.run(tf.global_variables_initializer())

    def _get_normalized_rewards(self):
        reward_normalized = np.zeros_like(self.r_buffer)
        reward_delta = 0
        for index in reversed(range(0, len(self.r_buffer))):
            reward_delta = reward_delta * self.gamma + self.r_buffer[index]
            reward_normalized[index] = reward_delta
        reward_normalized -= np.mean(reward_normalized)
        reward_normalized /= np.std(reward_normalized)
        return reward_normalized

    def get_next_action(self, state):
        action_prob = self.session.run(self.action_prob, feed_dict={self.state: state[np.newaxis, :]})
        return np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel())

    def save_transition(self, state, action, reward):
        self.s_buffer.append(state)
        self.a_buffer.append(action)
        self.r_buffer.append(reward)

    def train(self):

        reward_normalized = self._get_normalized_rewards()

        _, loss = self.session.run([self.train_op, self.loss], feed_dict={
            self.state: np.vstack(self.s_buffer),
            self.action: np.array(self.a_buffer),
            self.reward: reward_normalized,
        })

        self.train_steps += 1

        print("Train steps: {} | The loss is: {}".format(self.train_steps, loss))

        self.s_buffer, self.a_buffer, self.r_buffer = [], [], []


def run():
    env = gym.make('CartPole-v0')
    env.seed(1)
    env = env.unwrapped

    model = PolicyGradient(env.action_space.n, env.observation_space.shape[0])

    running_reward_sum = None

    for episode in range(3000):

        state = env.reset()

        while True:

            if episode > 80:
                env.render()

            action = model.get_next_action(state)

            state_next, reward, done, info = env.step(action)

            model.save_transition(state, action, reward)

            if done:
                if running_reward_sum is None:
                    running_reward_sum = sum(model.r_buffer)
                else:
                    running_reward_sum = running_reward_sum * 0.99 + sum(model.r_buffer) * 0.01

                model.train()

                break

            state = state_next


if __name__ == '__main__':
    run()
