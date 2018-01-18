# coding=utf-8

import threading
import logging
import random
import time

import tensorflow as tf
import numpy as np
import tflearn
import gym

from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

logging.basicConfig(level=logging.INFO)

IS_TEST = False

THREAD_COUNT = 8

GAME_NAME = "MsPacman-v0"

STEPS_MAX = 80000000

ACTION_FRAMES = 4

ASYNC_UPDATE_STEP = 4

TARGET_DQN_RESET_STEP = 40000

LEARNING_RATE = 0.001

REWARD_GAMMA = 0.99

ANNEAL_EPSILON_STEP = 40000


class DQN(object):
    def __init__(self, action_dim, frame_dim):
        self._init_session()
        self._init_input(action_dim, frame_dim)
        self._init_nn()
        self._init_ops()

    def _init_session(self):
        self.session = tf.Session()

    def _init_input(self, action_dim, frame_dim):
        self.action_dim, self.frame_dim = action_dim, frame_dim
        self.s_input = tf.placeholder(tf.float32, [None, self.frame_dim, 84, 84])
        self.a_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

    def _init_nn(self):
        self.state, self.q_values = self._build_dqn()
        self.state_t, self.target_q_values = self._build_dqn()

    def _init_ops(self):
        trainable_variables = tf.trainable_variables()
        self.dqn_params = trainable_variables[:int(len(trainable_variables) / 2)]
        self.target_dqn_params = trainable_variables[int(len(trainable_variables) / 2):]

        self.reset_target_dqn_params = []
        for index in range(len(self.target_dqn_params)):
            self.reset_target_dqn_params.append(self.target_dqn_params[index].assign(self.dqn_params[index]))

        self.action_q_values = tf.reduce_sum(tf.multiply(self.q_values, self.a_input), reduction_indices=1)

        self.loss = tf.reduce_sum(tf.square(self.action_q_values - self.y_input))

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss, var_list=self.dqn_params)

    def _build_dqn(self):
        s_transpose = tf.transpose(self.s_input, [0, 2, 3, 1])
        s_conv_1 = tflearn.conv_2d(s_transpose, 32, 8, strides=4, activation='relu')
        s_conv_2 = tflearn.conv_2d(s_conv_1, 64, 4, strides=2, activation='relu')
        s_fc = tflearn.fully_connected(s_conv_2, 256, activation='relu')
        q_values = tflearn.fully_connected(s_fc, self.action_dim)
        return s_transpose, q_values

    def actor_thread(self, thread_id, env):
        env = AtariEnv(env, self.frame_dim)
        state_batch, action_batch, y_batch = [], [], []
        initial_epsilon, final_epsilon, epsilon = 1.0, self.sample_final_epsilon(), 1.0

        logging.info("Thread: " + str(thread_id) + ", Final epsilon: " + str(final_epsilon))

        time.sleep(3 * thread_id)

        step = 0

        while step < STEPS_MAX:
            state_t, action_t = env.get_initial_state(), np.zeros([self.action_dim])
            episode_reward, episode_ave_max_q, episode_step = 0, 0, 0
            while True:

                q_values_result = self.q_values.eval(session=self.session,
                                                     feed_dict={self.s_input: [state_t]})

                if random.random() < epsilon:
                    action_index = random.randrange(self.action_dim)
                else:
                    action_index = np.argmax(q_values_result)
                action_t[action_index] = 1

                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / ANNEAL_EPSILON_STEP

                state_t1, reward_t, episode_done, info = env.get_next_step(action_index)

                target_q_values_result = self.target_q_values.eval(session=self.session,
                                                                   feed_dict={self.s_input: [state_t1]})

                clipped_reward_t = np.clip(reward_t, -1, 1)

                if episode_done:
                    y_batch.append(clipped_reward_t)
                else:
                    y_batch.append(clipped_reward_t + REWARD_GAMMA * np.max(target_q_values_result))

                action_batch.append(action_t)
                state_batch.append(state_t)

                state_t = state_t1

                step += 1
                episode_step += 1
                episode_reward += reward_t
                episode_ave_max_q += np.max(q_values_result)

                if step % TARGET_DQN_RESET_STEP == 0:
                    self.session.run(self.reset_target_dqn_params)

                if episode_step % ASYNC_UPDATE_STEP == 0 or episode_done:
                    if state_batch:
                        self.session.run(self.optimizer, feed_dict={self.a_input: action_batch,
                                                                    self.s_input: state_batch,
                                                                    self.y_input: y_batch})
                    state_batch, action_batch, y_batch = [], [], []

                if episode_done:
                    logging.info("| Thread %.2i" % int(thread_id))
                    logging.info("| Step %.2i" % episode_step)
                    logging.info("| Reward: %.2i" % int(episode_reward))
                    logging.info("| Qmax: %.4f" % (episode_ave_max_q / float(episode_step)))
                    logging.info("| Epsilon: %.5f" % epsilon)
                    logging.info("| Epsilon progress: %.6f" % (episode_step / float(ANNEAL_EPSILON_STEP)))

                    break

    def train(self):
        envs = [gym.make(GAME_NAME) for _ in range(THREAD_COUNT)]

        self.session.run(tf.global_variables_initializer())
        self.session.run(self.reset_target_dqn_params)

        actor_threads = []
        for thread_id in range(THREAD_COUNT):
            actor_threads.append(threading.Thread(target=self.actor_thread, args=(thread_id, envs[thread_id])))
        for thread in actor_threads:
            thread.start()
            time.sleep(0.05)
        while True:
            for env in envs:
                env.render()
        # for thread in actor_threads:
        #     thread.join()

    @staticmethod
    def sample_final_epsilon():
        final_epsilons, probabilities = np.array([0.1, 0.01, 0.5]), np.array([0.4, 0.3, 0.3])
        return np.random.choice(final_epsilons, 1, p=probabilities.tolist())[0]


class AtariEnv(object):
    def __init__(self, gym_env, frame_dim):
        self.env = gym_env
        self.frame_dim = frame_dim
        self.state_buffer = deque()
        self.action_dim_space = range(self.env.action_space.n)

    @staticmethod
    def get_preprocessed_frame(frame):
        # Origin frame is 210 * 160, grayscale it and resize to 110 * 84, then crop it to 84 * 84
        return resize(rgb2gray(frame), (110, 84))[13: 110 - 13, :]

    @classmethod
    def get_action_dim(cls):
        env = gym.make(GAME_NAME)
        return env.action_space.n

    def get_initial_state(self):
        self.state_buffer = deque()

        frame = self.env.reset()
        frame = self.get_preprocessed_frame(frame)

        state = np.stack([frame for _ in range(self.frame_dim)], axis=0)

        for frame_index in range(self.frame_dim - 1):
            self.state_buffer.append(frame)

        return state

    def get_next_step(self, action_index):
        frame, reward, episode_done, info = self.env.step(self.action_dim_space[action_index])
        frame = self.get_preprocessed_frame(frame)
        frame_previous = np.array(self.state_buffer)

        state_t = np.empty((self.frame_dim, 84, 84))
        state_t[:self.frame_dim - 1, :] = frame_previous
        state_t[self.frame_dim - 1] = frame

        self.state_buffer.popleft()
        self.state_buffer.append(frame)

        return state_t, reward, episode_done, info


def main(_):
    dqn = DQN(AtariEnv.get_action_dim(), 4)
    dqn.train()


if __name__ == '__main__':
    tf.app.run()
