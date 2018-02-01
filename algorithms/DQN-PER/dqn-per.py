# coding=utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import gym


logging.basicConfig(level=logging.INFO)


class SumTree(object):

    def __init__(self, t_data_capacity):
        self.t_data_pointer = 0
        self.t_data_capacity = t_data_capacity
        self.q_tree = np.zeros(2 * t_data_capacity - 1)
        self.t_data = np.zeros(t_data_capacity, dtype=object)

    def update_p_value(self, tree_index, p_value):
        diff = p_value - self.q_tree[tree_index]
        self.q_tree[tree_index] = p_value
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.q_tree[tree_index] += diff

    def add_p_value(self, p_value, t_data):
        tree_index = self.t_data_pointer + self.t_data_capacity - 1
        self.t_data[self.t_data_pointer] = t_data
        self.update_p_value(tree_index, p_value)
        self.t_data_pointer += 1
        if self.t_data_pointer >= self.t_data_capacity:
            self.t_data_pointer = 0

    def get_leaf(self, p_value):
        parent_index = 0
        while True:
            l_index = 2 * parent_index + 1
            r_index = l_index + 1
            if l_index >= len(self.q_tree):
                leaf_index = parent_index
                break
            else:
                if p_value <= self.q_tree[l_index]:
                    parent_index = l_index
                else:
                    p_value -= self.q_tree[l_index]
                    parent_index = r_index
        data_index = leaf_index - self.t_data_capacity + 1
        return leaf_index, self.q_tree[leaf_index], self.t_data[data_index]

    @property
    def total_p_value(self):
        return self.q_tree[0]


class Buffer(object):

    def __init__(self, capacity):
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_delta = 0.001
        self.p_value_bound = 1.0
        self.sum_tree = SumTree(capacity)

    def save_transition(self, transition):
        max_p_value = np.max(self.sum_tree.q_tree[-self.sum_tree.t_data_capacity:])
        if max_p_value == 0:
            max_p_value = self.p_value_bound
        self.sum_tree.add_p_value(max_p_value, transition)

    def sample_batch(self, batch_size):
        # 1. Update beta.
        self._update_beta()

        # 2. Initialize batch_indices and batch_transitions and batch_weights.
        batch_indices = np.empty((batch_size,), dtype=np.int32)
        batch_transitions = np.empty((batch_size, self.sum_tree.t_data[0].size))
        batch_weights = np.empty((batch_size, 1))

        # 3. Calculate sample segments.
        segments = self.sum_tree.total_p_value / batch_size

        # 4. Calculate min_p_value and min_prob
        min_p_value = np.min(self.sum_tree.q_tree[-self.sum_tree.t_data_capacity:])
        min_prob = min_p_value / self.sum_tree.total_p_value

        for batch_index in range(batch_size):

            # 5. Calculate left and right bound for each segment.
            low_bound, high_bound = segments * batch_index, segments * (batch_index + 1)

            # 6. Generate random p_value in each segment.
            random_p_value_in_segment = np.random.uniform(low_bound, high_bound)

            # 7. Get left_index and p_value and transition by random_p_value.
            leaf_index, p_value, transition = self.sum_tree.get_leaf(random_p_value_in_segment)

            # 8. Calculate prob, MORE P_VALUE, MORE PROB.
            prob = p_value / self.sum_tree.total_p_value

            # 9. Calculate weight for each sample.
            batch_weights[batch_index, 0] = np.power(prob / min_prob, -self.beta)

            # 10. Save sample tree index.
            batch_indices[batch_index] = leaf_index

            # 11. Save sample transition index.
            batch_transitions[batch_index, :] = transition

        return batch_indices, batch_transitions, batch_weights

    def update_batch(self, indices, q_value_diffs):

        # 1. Update q_value_diffs with epsilon as delta.
        q_value_diffs += self.epsilon

        # 2. Clip q_value_diffs by self.p_value_bound.
        q_value_diffs_clipped = np.minimum(q_value_diffs, self.p_value_bound)

        # 3. Zip indices with exp(q_value_clipped, self.alpha), THE MORE ALPHA, THE MORE EFFECT.
        q_value_diffs_zipped = zip(indices, np.power(q_value_diffs_clipped, self.alpha))

        for index, q_value_diff in q_value_diffs_zipped:
            # 4. Update sum tree with q_diff.
            self.sum_tree.update_p_value(index, q_value_diff)

    def _update_beta(self):
        self.beta = np.min([1.0, self.beta + self.beta_delta])


class DQN(object):

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

        try:
            self.reset_q_target_net_step = options['reset_q_target_net_step']
        except KeyError:
            self.reset_q_target_net_step = 500

        try:
            self.enable_PER = options['enable_PER']
        except KeyError:
            self.enable_PER = True

        try:
            self.session = options['session']
        except KeyError:
            self.session = tf.Session()

        self.action_dim = action_dim

        self.state_dim = state_dim

        self.total_steps = 0

        self.loss_history = []

        self.buffer_item_count = 0

        if self.enable_PER:
            self.per_buffer = Buffer(self.buffer_size)
        else:
            self.buffer = np.zeros((self.buffer_size, self.state_dim + 1 + 1 + self.state_dim))

        self._init_input()
        self._init_nn()
        self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        self.state_next = tf.placeholder(tf.float32, [None, self.state_dim], name='state_next')
        self.state_weight = tf.placeholder(tf.float32, [None, 1], name='state_weight')
        self.q_target_real = tf.placeholder(tf.float32, [None, self.action_dim], name='q_target_real')

    def _init_nn(self):
        self.q_predict = self.__build_q_net(self.state, 20, 'q_predict_net', True)
        self.q_target = self.__build_q_net(self.state_next, 20, 'q_target_net', False)

    def _init_op(self):

        with tf.variable_scope('loss'):
            if self.enable_PER:
                self.q_diffs = tf.reduce_sum(tf.abs(self.q_target_real - self.q_predict), axis=1)
                self.loss = tf.reduce_mean(self.state_weight * tf.squared_difference(self.q_target_real, self.q_predict))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target_real, self.q_predict))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('reset_q_target_net'):
            p_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='q_predict_net')
            t_params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target_net')
            self.reset_q_target_net = [tf.assign(t, p) for t, p in zip(t_params, p_params)]

        self.session.run(tf.global_variables_initializer())

    def _reset_target_q_net_if_need(self):
        if self.total_steps % self.reset_q_target_net_step == 0:
            self.session.run(self.reset_q_target_net)
            logging.info("Steps: {} | Q-target Reset.".format(self.total_steps))

    def __build_q_net(self, state, unit_count, scope_name, is_trainable):

        w_initializer, b_initializer = tf.random_normal_initializer(.0, .3), tf.constant_initializer(0.1)

        with tf.variable_scope(scope_name):

            phi_state = tf.layers.dense(state,
                                        unit_count,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer,
                                        trainable=is_trainable)

            q_values = tf.layers.dense(phi_state,
                                       self.action_dim,
                                       kernel_initializer=w_initializer,
                                       bias_initializer=b_initializer,
                                       trainable=is_trainable)
            return q_values

    def save_transition(self, state, action, reward, state_next):
        transition = np.hstack((state, [action, reward], state_next))
        if self.enable_PER:
            self.per_buffer.save_transition(transition)
        else:
            index = self.buffer_item_count % self.buffer_size
            self.buffer[index, :] = transition
            self.buffer_item_count += 1

    def get_next_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.argmax(self.session.run(self.q_predict, feed_dict={self.state: state[np.newaxis, :]}))
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def train(self):
        self._reset_target_q_net_if_need()

        if self.enable_PER:
            indices, batch, state_weight = self.per_buffer.sample_batch(self.batch_size)
        else:
            batch = self.buffer[np.random.choice(self.buffer_size, size=self.batch_size), :]

        state = batch[:, :self.state_dim]
        action = batch[:, self.state_dim].astype(int)
        reward = batch[:, self.state_dim + 1]
        state_next = batch[:, -self.state_dim:]

        q_target, q_predict = self.session.run([self.q_target, self.q_predict], feed_dict={
            self.state_next: state_next, self.state: state
        })

        q_real = q_predict.copy()
        q_real[np.arange(self.batch_size, dtype=np.int32), action] = reward + self.gamma * np.max(q_target, axis=1)

        if self.enable_PER:
            _, q_values_diffs, loss = self.session.run([self.train_op, self.q_diffs, self.loss], feed_dict={
                self.state: state, self.q_target_real: q_real, self.state_weight: state_weight
            })
        else:
            _, loss = self.session.run([self.train_op, self.loss], feed_dict={
                self.state: state, self.q_target_real: q_real
            })

        if self.total_steps % 1000 == 0:
            logging.info("Steps: {} | The loss is: {}".format(self.total_steps, loss))

        self.loss_history.append(loss)

        self.total_steps += 1

    def run(self, env):

        steps, total_steps, episodes = [], 0, []

        for episode in range(20):

            state = env.reset()

            while True:

                # if total_steps > self.buffer_size:
                    # env.render()

                action = self.get_next_action(state)

                state_next, reward, done, info = env.step(action)

                if done:
                    reward = 10

                self.save_transition(state, action, reward, state_next)

                if total_steps > self.buffer_size:
                    self.train()

                if done:
                    steps.append(total_steps), episodes.append(episode)
                    logging.info("Run Steps: {} | Episode: {} | Finished.".format(total_steps, episode))
                    break

                state = state_next

                total_steps += 1

        return np.vstack((episodes, steps))


def main(_):

    _env = gym.make('MountainCar-v0')
    _env = _env.unwrapped
    _env.seed(25)

    session = tf.Session()

    with tf.variable_scope('DQN-PER'):
        model_per = DQN(3, 2, session=session, buffer_size=10000, enable_PER=True)
        info_per = model_per.run(_env)

    with tf.variable_scope('DQN'):
        model = DQN(3, 2, session=session, buffer_size=10000, enable_PER=False)
        info = model.run(_env)

    plt.plot(info[0, :], info[1, :] - info[1, 0], c='b', label='natural DQN')
    plt.plot(info_per[0, :], info_per[1, :] - info_per[1, 0], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    tf.app.run()
