# coding=utf-8

import tensorflow as tf
import numpy as np
import gym


class Actor(object):

    def __init__(self, session, state_space, action_bounds, **options):

        self.session = session

        self.state_space = state_space

        self.action_bounds = action_bounds

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.0001

        self.global_steps = tf.Variable(0, trainable=False)

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [1, self.state_space])
        self.action = tf.placeholder(tf.float32, None)
        self.td_error = tf.placeholder(tf.float32, None)

    def _init_nn(self):
        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        phi_state = tf.layers.dense(self.state,
                                    20,
                                    tf.nn.relu,
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init)

        mu = tf.layers.dense(phi_state,
                             1,
                             tf.nn.tanh,
                             kernel_initializer=w_init,
                             bias_initializer=b_init)

        sigma = tf.layers.dense(phi_state,
                                1,
                                tf.nn.softplus,
                                kernel_initializer=w_init,
                                bias_initializer=b_init)

        self.mu, self.sigma = tf.squeeze(mu * 2), tf.squeeze(sigma + 0.1)

        self.gaussian_func = tf.distributions.Normal(self.mu, self.sigma)

        gaussian_sample = self.gaussian_func.sample(sample_shape=1)

        self.action = tf.clip_by_value(gaussian_sample, self.action_bounds[0], self.action_bounds[1])

    def _init_op(self):

        self.loss = self.gaussian_func.log_prob(self.action) * self.td_error + 0.01 * self.gaussian_func.entropy()

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.loss, self.global_steps)

    def train(self, state, action, td_error):
        _, loss = self.session.run([self.train_op, self.loss], feed_dict={
            self.state: state[np.newaxis, :], self.action: action, self.td_error: td_error
        })
        return loss

    def get_next_action(self, state):
        return self.session.run(self.action, feed_dict={self.state: state[np.newaxis, :]})


class Critic(object):

    def __init__(self, session, state_space, **options):

        self.session = session

        self.state_space = state_space

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.0001

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [1, self.state_space])
        self.reward = tf.placeholder(tf.float32)
        self.value_next = tf.placeholder(tf.float32, [1, 1])

    def _init_nn(self):
        w_init, b_init = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        phi_state = tf.layers.dense(self.state,
                                    30,
                                    tf.nn.relu,
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init)

        self.value = tf.layers.dense(phi_state,
                                     1,
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init)

    def _init_op(self):

        self.td_error = tf.reduce_mean(self.reward + self.gamma * self.value_next - self.value)

        self.loss = tf.square(self.td_error)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, state, reward, state_next):

        state, state_next = state[np.newaxis, :], state_next[np.newaxis, :]

        value_next = self.session.run(self.value, feed_dict={self.state: state})

        td_error, _ = self.session.run([self.td_error, self.train_op], feed_dict={
            self.state: state, self.value_next: value_next, self.reward: reward
        })

        return td_error


def main(_):
    env = gym.make('Pendulum-v0')
    env.seed(1)
    env = env.unwrapped

    session = tf.Session()

    actor = Actor(session, env.observation_space.shape[0], [-env.action_space.high,  env.action_space.high])

    critic = Critic(session, env.observation_space.shape[0])

    session.run(tf.global_variables_initializer())

    running_reward = None

    for episode in range(1000):

        state, steps, reward_history = env.reset(), 0, []

        while True:

            if episode > 500:

                env.render()

            action = actor.get_next_action(state)

            state_next, reward, done, info = env.step(action)

            reward /= 10

            td_error = critic.train(state, reward, state_next)

            actor.train(state, action, td_error)

            state = state_next

            steps += 1

            reward_history.append(reward)

            if steps > 200:
                if running_reward is None:
                    running_reward = sum(reward_history)
                else:
                    running_reward = running_reward * 0.9 + sum(reward_history) * 0.1
                print("Episode: {} | Reward: {}".format(episode, running_reward))
                break


if __name__ == '__main__':
    tf.app.run()