# coding=utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym


tf.set_random_seed(1)
np.random.seed(1)


class DuelingDQN(object):

    def __init__(self, action_dim, state_dim, **options):

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.001

        try:
            self.gamma = options['gamma']
        except KeyError:
            self.gamma = 0.9

        try:
            self.epsilon = options['epsilon']
        except KeyError:
            self.epsilon = 0.9

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 3000

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.update_q_target_net_step = options['update_q_target_net_step']
        except KeyError:
            self.update_q_target_net_step = 200

        try:
            self.enable_dueling = options['dueling']
        except KeyError:
            self.enable_dueling = True

        try:
            self.session = options['session']
        except KeyError:
            self.session = tf.Session()

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.train_steps = 0

        self.buffer = np.zeros((self.buffer_size, self.state_dim + 1 + 1 + self.state_dim))

        self.buffer_item_count = 0

        self.loss_history = []

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        self.state_next = tf.placeholder(tf.float32, [None, self.state_dim])
        self.q_value_real = tf.placeholder(tf.float32, [None, self.action_dim])

    def _init_nn(self):
        self.q_predict = self.__build_dqn(self.state, 20, scope_name="Q-predict")
        self.q_target = self.__build_dqn(self.state_next, 20, scope_name="Q-target")

    def _init_op(self):

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value_real, self.q_predict))

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope('update_Q_target_net'):
            prefix = "dueling" if self.enable_dueling else 'natural'
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "{}/Q-target".format(prefix))
            self.p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "{}/Q-predict".format(prefix))
            self.update_q_target_net = [tf.assign(t, e) for t, e in zip(self.t_params, self.p_params)]

        self.session.run(tf.global_variables_initializer())

    def __build_dqn(self, state, unit_count, scope_name):

        w_initializer, b_initializer = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        if self.enable_dueling:

            with tf.variable_scope(scope_name):

                phi_state = tf.layers.dense(state,
                                            unit_count,
                                            activation=tf.nn.relu,
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer)

                self.value = tf.layers.dense(phi_state,
                                             1,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer)

                self.advantage = tf.layers.dense(phi_state,
                                                 self.action_dim,
                                                 kernel_initializer=w_initializer,
                                                 bias_initializer=b_initializer)

                q_value = self.value + (self.advantage - tf.reduce_mean(self.advantage, axis=1, keep_dims=True))

                return q_value

        else:

            with tf.variable_scope(scope_name):

                phi_state = tf.layers.dense(state,
                                            unit_count,
                                            activation=tf.nn.relu,
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer)

                q_value = tf.layers.dense(phi_state,
                                          self.action_dim,
                                          kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer)

                return q_value

    def save_transition(self, state, action, reward, state_next):
        self.buffer[self.buffer_item_count % self.buffer_size, :] = np.hstack((state, [action, reward], state_next))
        self.buffer_item_count += 1

    def get_next_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.argmax(self.session.run(self.q_predict, feed_dict={self.state: state[np.newaxis, :]}))
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def update_q_target_net_if_need(self):
        if self.train_steps % self.update_q_target_net_step == 0:
            self.session.run(self.update_q_target_net)
            print('Train Steps: {} | Q-Target Net updated.'.format(self.train_steps))

    def sample_batch(self):
        batch = self.buffer[np.random.choice(self.buffer_size, size=self.batch_size), :]

        state = batch[:, :self.state_dim]
        state_next = batch[:, -self.state_dim:]
        action, reward = batch[:, self.state_dim].astype(int), batch[:, self.state_dim + 1]

        return batch, state, action, reward, state_next

    def train(self):

        self.update_q_target_net_if_need()

        batch, state, action, reward, state_next = self.sample_batch()

        q_value_predict, q_value_target = self.session.run([self.q_predict, self.q_target], feed_dict={
            self.state: state, self.state_next: state_next
        })

        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        q_value_real = q_value_predict.copy()
        q_value_real[batch_indices, action] = reward + self.gamma * np.max(q_value_target, axis=1)

        _, loss = self.session.run([self.train_op, self.loss], feed_dict={
            self.state: state, self.q_value_real: q_value_real
        })

        if self.train_steps % 1000 == 0:
            print('Train Steps: {} | Loss is {}'.format(self.train_steps, loss))

        self.loss_history.append(loss)
        self.train_steps += 1

    def run(self, env):

        rewards, total_steps, state = [0], 0, env.reset()

        while True:

            if total_steps - self.buffer_size > 10000:
                env.render()

            action = self.get_next_action(state)

            action_normalized = (action - (self.action_dim - 1) / 2) / ((self.action_dim - 1) / 4)

            state_next, reward, done, info = env.step(np.array([action_normalized]))

            reward /= 10

            rewards.append(reward + rewards[-1])

            self.save_transition(state, action, reward, state_next)

            if total_steps > self.buffer_size:
                self.train()

            if total_steps - self.buffer_size > 15000:
                break

            state = state_next

            total_steps += 1

        return self.loss_history, rewards


def main(_):

    _env = gym.make('Pendulum-v0')
    _env = _env.unwrapped
    _env.seed(1)

    session = tf.Session()
    with tf.variable_scope('natural'):
        natural_dqn = DuelingDQN(25, 3, buffer_size=3000, dueling=False, session=session)

    with tf.variable_scope('dueling'):
        dueling_dqn = DuelingDQN(25, 3, buffer_size=3000, dueling=True, session=session)

    c_natural, r_natural = natural_dqn.run(_env)
    c_dueling, r_dueling = dueling_dqn.run(_env)

    plt.figure(1)
    plt.plot(np.array(c_natural), c='r', label='natural')
    plt.plot(np.array(c_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('cost')
    plt.xlabel('training steps')
    plt.grid()

    plt.figure(2)
    plt.plot(np.array(r_natural), c='r', label='natural')
    plt.plot(np.array(r_dueling), c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    tf.app.run()
