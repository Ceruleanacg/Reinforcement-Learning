# coding=utf-8

import tensorflow as tf
import numpy as np
import gym


class DoubleDQN(object):
    def __init__(self, action_dim, state_dim, **options):

        try:
            self.learning_rate = options['learning_rate']
        except KeyError:
            self.learning_rate = 0.005

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

        # Initialize parameters.
        self.action_dim, self.state_dim = action_dim, state_dim
        self.buffer = np.zeros((self.buffer_size, self.state_dim + 1 + 1 + self.state_dim))
        self.buffer_item_count = 0
        self.q_history = []
        self.q_running = 0
        self.total_steps = 0
        self.update_q_net_step = 200
        self.loss_history = []

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_session()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        self.state_next = tf.placeholder(tf.float32, [None, self.state_dim], name='state_next')
        self.q_target = tf.placeholder(tf.float32, [None, self.action_dim])

    def _init_nn(self):

        w_initializer, b_initializer = tf.random_normal_initializer(0.0, 0.3), tf.constant_initializer(0.1)

        self.q_values_target = self.__build_layers(self.state_next, 20, w_initializer, b_initializer, "target_qn")
        self.q_values_predict = self.__build_layers(self.state, 20, w_initializer, b_initializer, "predict_qn")

    def _init_op(self):
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_values_predict))
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('update_target_q_net'):
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_qn')
            self.p_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='predict_qn')
            self.update_target_q_net_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.p_params)]

    def _init_session(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __build_layers(self, state, unit_count, w_initializer, b_initializer, scope):
        with tf.variable_scope(scope):
            phi_state = tf.layers.dense(state,
                                        unit_count,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer)

            q_values = tf.layers.dense(phi_state,
                                       self.action_dim,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer)
            return q_values

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, [action, reward], state_next))
        index = self.buffer_item_count % self.buffer_size
        self.buffer[index, :] = transition
        self.buffer_item_count += 1

    def get_next_action(self, state):
        if np.random.uniform() < self.epsilon:
            # Calculate Q-values Predict.
            q_predict = self.session.run(self.q_values_predict, feed_dict={self.state: state[np.newaxis, :]})
            # Get target action form max(Q-values)
            target_action = np.argmax(q_predict)
            # Update Q-running.
            self.q_running = self.q_running * 0.99 + 0.01 * np.max(q_predict)
            self.q_history.append(self.q_running)
        else:
            target_action = np.random.randint(0, self.action_dim)
        return target_action

    def get_sample_batch(self):
        if self.buffer_item_count > self.buffer_size:
            sample_indices = np.random.choice(self.buffer_size, size=self.batch_size)
        else:
            sample_indices = np.random.choice(self.buffer_item_count, size=self.batch_size)
        return self.buffer[sample_indices, :]

    def update_target_q_net_if_need(self):
        if self.total_steps % self.update_q_net_step == 0:
            self.session.run(self.update_target_q_net_op)
            print('Steps:{} | Target Q-network has updated.'.format(self.total_steps))

    def train(self):
        # 1. Update target Q-network if need.
        self.update_target_q_net_if_need()

        # 2. Get batch sample and batch indices.
        batch = self.get_sample_batch()
        action = batch[:, self.state_dim].astype(int)
        state = batch[:, :self.state_dim]
        state_next = batch[:, -self.state_dim:]
        reward = batch[:, self.state_dim + 1]

        # 3-1. Calculate Q-predict (s')
        q_predict_next = self.session.run(self.q_values_predict, feed_dict={self.state: state_next})

        # 3-2. Get argmax action indices, for Q-predict (s') and create batch indices.
        argmax_actions_indices = np.argmax(q_predict_next, axis=1)
        batch_indices = np.arange(self.batch_size, dtype=np.int32)

        # 3-3. Calculate Q-target (s', max_a)
        q_target_next = self.session.run(self.q_values_target, feed_dict={self.state_next: state_next})
        q_target_next_max_a = q_target_next[batch_indices, argmax_actions_indices]

        # 3-4. Calculate y_i
        q_predict = self.session.run(self.q_values_predict, feed_dict={self.state: state})
        q_target = q_predict.copy()
        q_target[batch_indices, action] = reward + self.gamma * q_target_next_max_a

        # 6. Calculate loss by calculate Q-predict (s, a).
        _, loss = self.session.run([self.train_op, self.loss], feed_dict={self.state: state, self.q_target: q_target})

        # 7. Logs.
        self.loss_history.append(loss)
        self.total_steps += 1

        print('Steps: {}, the loss is :{}'.format(self.total_steps, loss))


ACTION_DIM, STATE_DIM = 11, 3
# ACTION_DIM, STATE_DIM = 3, 2


def main(_):

    env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCar-v0')
    env = env.unwrapped
    env.seed(2)

    model = DoubleDQN(ACTION_DIM, STATE_DIM)

    state = env.reset()

    total_steps = 0

    while True:

        if total_steps > 8000:
            env.render()

        action = model.get_next_action(state)
        action_normalized = (action - (ACTION_DIM - 1) / 2) / ((ACTION_DIM - 1) / 4)

        state_next, reward, done, info = env.step(np.array([action_normalized]))

        reward /= 10.0

        model.save_transition(state, action, reward, state_next)

        if total_steps > 2000:
            model.train()

        if total_steps > 2000 + 50000:
            break

        state = state_next
        total_steps += 1


if __name__ == '__main__':
    tf.app.run()
