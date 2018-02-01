# coding=utf-8

import pandas as pd
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO)

np.random.seed(2)


class QLearning(object):
    def __init__(self, state_count, actions, epsilon=0.9, alpha=0.1, gamma=0.9, max_episodes=20):

        self.state_count = state_count
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.current_state = 0
        self.current_episode = 0
        self.current_steps = 0

        self._init_q_table()

    def _init_q_table(self):
        self.q_table = pd.DataFrame(np.zeros((self.state_count, len(self.actions))), columns=self.actions)

    def get_action(self):
        target_actions = self.q_table.iloc[self.current_state, :]
        if np.random.uniform() > self.epsilon or target_actions.all() == 0:
            target_action = np.random.choice(self.actions)
        else:
            target_action = target_actions.idxmax()
        return target_action

    def get_next_state(self, action):
        if action == 'right':
            if self.current_state + 2 == self.state_count:
                state_next = 'episode_done'
                reward = 1
            else:
                state_next = self.current_state + 1
                reward = 0
        else:
            reward = -1
            if self.current_state == 0:
                state_next = 0
            else:
                state_next = self.current_state - 1
        return state_next, reward

    def update_env(self):
        env = ['-'] * (self.state_count - 1) + ['💰']
        if self.current_state == 'episode_done':
            logging.info("Episode %s, total steps = %s" % (self.current_episode, self.current_steps))
            logging.info("\r{}".format(self.q_table))
            time.sleep(1.5)
        else:
            env[self.current_state] = '👶'
            logging.info("".join(env))
            time.sleep(0.3)

    def train(self):
        for episode in range(self.max_episodes):

            self.current_state = 0
            self.current_steps = 0

            self.update_env()

            while True:

                action = self.get_action()
                state_next, reward = self.get_next_state(action)

                q_value_predict = self.q_table.loc[self.current_state, action]

                if state_next == 'episode_done':
                    episode_done = True
                else:
                    episode_done = False

                if not episode_done:
                    q_value_target = reward + self.gamma * self.q_table.iloc[state_next, :].max()
                else:
                    q_value_target = reward

                self.q_table.loc[self.current_state, action] += self.alpha * (q_value_target - q_value_predict)
                self.current_state = state_next
                self.current_steps += 1

                self.update_env()

                if episode_done:
                    break


if __name__ == '__main__':
    model = QLearning(6, ['left', 'right'])
    model.train()
