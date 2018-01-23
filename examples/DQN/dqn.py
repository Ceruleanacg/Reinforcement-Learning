# coding=utf-8

import tensorflow as tf
import numpy as np

from base.maze import Maze

np.random.seed(1)
tf.set_random_seed(1)


class DQN(object):
    def __init__(self, env, action_dim, state_dim, **options):

        self.env = env

        # Init action dim.
        self.action_dim = action_dim

        # Init state dim.
        self.state_dim = state_dim

        # Init cost history.
        self.cost_history = []

        # Init buffer count.
        self.buffer_count = 0

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
            self.reset_steps = 200

        try:
            self.buffer_size = options['buffer_size']
        except KeyError:
            self.buffer_size = 2000

        try:
            self.batch_size = options['batch_size']
        except KeyError:
            self.batch_size = 32

        try:
            self.need_save_graph = options['need_save_graph']
        except KeyError:
            self.need_save_graph = False

        self.total_steps = 0

        self.buffer = np.zeros((self.buffer_size, state_dim + 1 + 1 + state_dim))

        self._init_inputs()
        self._init_nn()
        self._init_ops()
        self._init_session()

    def _init_inputs(self):
        # Input state, state_next, reward, action.
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='input_state')
        self.state_next = tf.placeholder(tf.float32, [None, self.state_dim], name='input_state_next')
        self.reward = tf.placeholder(tf.float32, [None, ], name='input_reward')
        self.action = tf.placeholder(tf.int32, [None, ], name='input_action')

    def _init_nn(self):

        # w,b initializer
        w_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.3)
        b_initializer = tf.constant_initializer(0.1)

        # Evaluate net.
        with tf.variable_scope('predict_q_net'):
            phi_state = tf.layers.dense(self.state,
                                        20,
                                        tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        name='phi_state_fc')

            self.q_values_predict = tf.layers.dense(phi_state,
                                                    self.action_dim,
                                                    kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer,
                                                    name='Q_predict')

        with tf.variable_scope('target_q_net'):
            phi_state_next = tf.layers.dense(self.state_next,
                                             20,
                                             tf.nn.relu,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name='phi_state_next_fc')

            self.q_values_target = tf.layers.dense(phi_state_next,
                                                   self.action_dim,
                                                   kernel_initializer=w_initializer,
                                                   bias_initializer=b_initializer,
                                                   name='Q_target')

    def _init_ops(self):
        with tf.variable_scope('q_real'):
            # size of q_value_real is [BATCH_SIZE, 1]
            q_value_max = tf.reduce_max(self.q_values_target, axis=1)
            q_value_real = self.reward + self.gamma * q_value_max
            self.q_value_real = tf.stop_gradient(q_value_real)

        with tf.variable_scope('q_predict'):
            # size of q_value_predict is [BATCH_SIZE, 1]
            action_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            self.q_value_predict = tf.gather_nd(self.q_values_predict, action_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value_real, self.q_value_predict, name='mse'))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)

        with tf.variable_scope('update_target_net'):
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_net')
            p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict_q_net')

            self.update_q_net = [tf.assign(t, e) for t, e in zip(t_params, p_params)]

    def _init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if self.need_save_graph:
            tf.summary.FileWriter('logs/', self.session.graph)

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_count % self.buffer_size
        self.buffer[index, :] = transition
        self.buffer_count += 1

    def get_next_action(self, state):
        state = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            target_action = np.argmax(self.session.run(self.q_values_predict, feed_dict={self.state: state}))
        else:
            target_action = np.random.randint(0, self.action_dim)
        return target_action

    def train(self):

        game_step = 0

        for episode in range(300):
            state = self.env.reset()

            while True:

                self.env.render()

                action = self.get_next_action(state)

                state_next, reward, done = self.env.step(action)

                self.save_transition(state, action, reward, state_next)

                if game_step > 200 and game_step % 5 == 0:

                    if self.total_steps % self.reset_steps == 0:
                        self.session.run(self.update_q_net)
                        print('Target Q network updated.')
                    if self.buffer_count > self.buffer_size:
                        sample_indices = np.random.choice(self.buffer_size, size=self.batch_size)
                    else:
                        sample_indices = np.random.choice(self.buffer_count, size=self.batch_size)

                    batch = self.buffer[sample_indices, :]

                    batch_s = batch[:, :self.state_dim]
                    batch_a = batch[:, self.state_dim]
                    batch_r = batch[:, self.state_dim + 1]
                    batch_s_n = batch[:, -self.state_dim:]

                    _, cost = self.session.run(
                        [self.train_op, self.loss],
                        feed_dict={
                            self.state: batch_s,
                            self.action: batch_a,
                            self.reward: batch_r,
                            self.state_next: batch_s_n
                        }
                    )

                    print('The MSE is: {:.4}, total steps is: {}'.format(float(cost), self.total_steps))

                    self.cost_history.append(cost)
                    self.total_steps += 1

                state = state_next

                if done:
                    break

                game_step += 1
        print('Game Over')
        self.env.destroy()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()


if __name__ == '__main__':

    # Init Maze, env.
    _env = Maze()

    # Init Model.
    model = DQN(_env, _env.n_actions, _env.n_features)

    # Train.
    _env.after(100, model.train)
    _env.mainloop()

    # Plot cost.
    model.plot_cost()
