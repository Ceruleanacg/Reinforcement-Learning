# coding=utf-8

import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing
import numpy as np
import threading
import shutil
import gym
import os


ENV_NAME = "CartPole-v0"
GLOBAL_ENV = gym.make(ENV_NAME)

STATE_SPACE, ACTION_SPACE = GLOBAL_ENV.observation_space.shape[0], GLOBAL_ENV.action_space.n

GLOBAL_EPISODE = 0
GLOBAL_EPISODE_MAX = 1000
GLOBAL_RUNNING_REWARD = []
GLOBAL_UPDATE_ITERATION = 10


class A3C(object):

    def __init__(self, session, state_space, action_space, scope, master_model=None):

        self.session = session

        self.state_space = state_space
        self.action_space = action_space

        self.master_model = master_model

        self.scope = scope

        with tf.variable_scope(self.scope):
            self._init_input()
            self._init_nn()
            self._init_op()

    def _init_input(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_space])
        self.action = tf.placeholder(tf.int32, [None, ])
        self.q_target = tf.placeholder(tf.float32, [None, 1])

    def _init_nn(self):

        w_init, b_init = tf.random_normal_initializer(.0, .1), tf.constant_initializer(.1)

        with tf.variable_scope("actor"):

            phi_state = tf.layers.dense(self.state,
                                        200,
                                        tf.nn.relu6,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init)

            self.action_prob = tf.layers.dense(phi_state,
                                               self.action_space,
                                               tf.nn.softmax,
                                               kernel_initializer=w_init,
                                               bias_initializer=b_init)

        with tf.variable_scope("critic"):

            phi_state = tf.layers.dense(self.state,
                                        100,
                                        tf.nn.relu6,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init)

            self.q_predict = tf.layers.dense(phi_state,
                                             1,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init)

        self.actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')

    def _init_op(self):

        if self.scope == 'master':

            self.actor_optimizer = tf.train.RMSPropOptimizer(0.001)
            self.critic_optimizer = tf.train.RMSPropOptimizer(0.002)

        else:

            self.td_error = tf.subtract(self.q_target, self.q_predict)

            with tf.variable_scope('critic_loss'):
                self.critic_loss = tf.reduce_mean(tf.square(self.td_error))

            with tf.variable_scope('actor_loss'):
                action_one_hot = tf.one_hot(self.action, self.action_space, dtype=tf.float32)
                minus_cross_entropy = -tf.reduce_sum(tf.log(self.action_prob) * action_one_hot, axis=1, keep_dims=True)
                self.actor_loss = tf.reduce_mean(minus_cross_entropy * tf.stop_gradient(self.td_error))

            with tf.variable_scope('local_gradients'):
                self.actor_gradients = tf.gradients(self.actor_loss, self.actor_params)
                self.critic_gradients = tf.gradients(self.critic_loss, self.critic_params)

            with tf.variable_scope('sync_pull'):

                zipped_actor_params = zip(self.master_model.actor_params, self.actor_params)
                zipped_critic_params = zip(self.master_model.critic_params, self.critic_params)

                self.pull_actor_params_op = [l_a_p.assign(g_a_p) for g_a_p, l_a_p in zipped_actor_params]
                self.pull_critic_params_op = [l_c_p.assign(g_c_p) for g_c_p, l_c_p in zipped_critic_params]

            with tf.variable_scope('sync_push'):
                zipped_actor_vars = zip(self.actor_gradients, self.master_model.actor_params)
                zipped_critic_vars = zip(self.critic_gradients, self.master_model.critic_params)
                self.update_actor_op = self.master_model.actor_optimizer.apply_gradients(zipped_actor_vars)
                self.update_critic_op = self.master_model.critic_optimizer.apply_gradients(zipped_critic_vars)

    def update_master_nn(self, feed_dict):
        self.session.run([self.update_actor_op, self.update_critic_op], feed_dict)

    def pull_master_nn(self):
        self.session.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def get_next_action(self, state):
        action_prob = self.session.run(self.action_prob, {self.state: state[np.newaxis, :]})
        action = np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel())
        return action


class Worker(object):

    def __init__(self, session, name, coordinator, master_model):
        self.env = gym.make(ENV_NAME).unwrapped
        self.name = name
        self.model = A3C(session, STATE_SPACE, ACTION_SPACE, 'slave-' + name, master_model=master_model)
        self.session = session
        self.coordinator = coordinator
        self.master_model = master_model

    def work(self):

        global GLOBAL_EPISODE

        buffer_action, buffer_state, buffer_reward, total_steps = [], [], [], 1

        while not self.coordinator.should_stop() and GLOBAL_EPISODE < GLOBAL_EPISODE_MAX:

            state = self.env.reset()

            reward_episode = 0

            while True:

                action = self.model.get_next_action(state)

                state_next, reward, done, info = self.env.step(action)

                if done:
                    reward = -5

                reward_episode += reward

                buffer_action.append(action)
                buffer_reward.append(reward)
                buffer_state.append(state)

                if total_steps % GLOBAL_UPDATE_ITERATION == 0 or done:

                    if done:
                        q_target = 0
                    else:
                        q_target = self.session.run(self.model.q_predict, {self.model.state: state_next[np.newaxis, :]})[0, 0]

                    buffer_q_target = []

                    for reward in buffer_reward[::-1]:
                        q_target = reward + 0.9 * q_target
                        buffer_q_target.append(q_target)

                    buffer_q_target.reverse()

                    feed_dict = {
                        self.model.state: np.vstack(buffer_state),
                        self.model.action: np.array(buffer_action),
                        self.model.q_target: np.vstack(buffer_q_target)
                    }

                    self.model.update_master_nn(feed_dict)

                    buffer_action, buffer_state, buffer_reward = [], [], []

                    self.model.pull_master_nn()

                state = state_next

                total_steps += 1

                if done:
                    if len(GLOBAL_RUNNING_REWARD) == 0:
                        GLOBAL_RUNNING_REWARD.append(reward_episode)
                    else:
                        GLOBAL_RUNNING_REWARD.append(0.99 * GLOBAL_RUNNING_REWARD[-1] + 0.01 * reward_episode)
                    print("Worker Name: {}| Episode: {}, Rewards: {}".format(self.name,
                                                                             GLOBAL_EPISODE,
                                                                             GLOBAL_RUNNING_REWARD[-1]))
                    GLOBAL_EPISODE += 1
                    break


def main(_):

    session = tf.Session()

    master_model = A3C(session, STATE_SPACE, ACTION_SPACE, "master")

    workers, coordinator = [], tf.train.Coordinator()

    # for index in range(1):
    for index in range(multiprocessing.cpu_count()):
        workers.append(Worker(session, "{}".format(index), coordinator, master_model))

    session.run(tf.global_variables_initializer())

    worker_threads = []

    for worker in workers:
        thread = threading.Thread(target=worker.work)
        thread.start()
        worker_threads.append(thread)

    coordinator.join(worker_threads)


if __name__ == '__main__':
    tf.app.run()
