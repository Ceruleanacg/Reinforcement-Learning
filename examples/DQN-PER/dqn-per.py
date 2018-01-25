# coding=utf-8

import tensorflow as tf
import numpy as np
import gym


class SumTree(object):

    def __init__(self, t_data_capacity):
        self.t_data_pointer = 0
        self.t_data_capacity = t_data_capacity
        self.q_tree = np.zeros(2 * t_data_capacity - 1)
        self.t_data = np.zeros(t_data_capacity, dtype=object)

    def update_p_value(self, index, p_value):
        diff = p_value - self.q_tree[index]
        self.q_tree[index] = diff
        while index != 0:
            index = (index - 1) // 2
            self.q_tree[index] += diff

    def add_p_value(self, p_value, t_data):
        index = self.t_data_pointer + self.t_data_capacity - 1
        self.t_data[self.t_data_pointer] = t_data
        self.update_p_value(index, p_value)
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
        data_index = leaf_index - self.t_data_pointer + 1
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
        self.q_value_bound = 1.0
        self.sum_tree = SumTree(capacity)

    def save_transition(self, transition):
        max_p_value = np.max(self.sum_tree.q_tree[-self.sum_tree.t_data_capacity:])
        if max_p_value == 0:
            max_p_value = self.q_value_bound
        self.sum_tree.add_p_value(max_p_value, transition)

    def sample_batch(self, batch_size):
        self._update_beta()

        batch_indices = np.empty((batch_size,), dtype=np.int32)
        batch_transitions = np.empty((batch_size, self.sum_tree.t_data[0].size))

        weights = np.empty((batch_size, 1))

        segments = self.sum_tree.total_p_value / batch_size

        min_p_value = np.min(self.sum_tree.q_tree[-self.sum_tree.t_data_capacity:])
        min_prob = min_p_value / self.sum_tree.total_p_value

        for index in range(batch_size):
            random_p_value_in_segment = np.random.uniform(segments * index, segments * (index + 1))
            index, p_value, transition = self.sum_tree.get_leaf(random_p_value_in_segment)
            prob = p_value / self.sum_tree.total_p_value
            weights[index, 0] = np.power(prob / min_prob, -self.beta)
            batch_indices[index], batch_transitions[index, :] = index, transition
        return batch_indices, batch_transitions, weights

    def update_batch(self, index, q_value_bound):
        q_value_bound += self.epsilon
        q_value_clipped = np.minimum(q_value_bound, self.q_value_bound)
        q_value_zipped = zip(index, np.power(q_value_clipped, self.alpha))

        for index, q_value in q_value_zipped:
            self.sum_tree.update_p_value(index, q_value)

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

        self.action_dim = action_dim

        self.state_dim = state_dim

        self.total_steps = 0


