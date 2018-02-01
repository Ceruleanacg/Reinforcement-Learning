# coding=utf-8

import pandas as pd
import numpy as np

from base.maze import Maze


class QLearning(object):
    def __init__(self, actions, env, alpha=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def get_action(self, state):
        self.check_if_state_exist(state)
        if np.random.uniform() < self.epsilon:
            target_actions = self.q_table.loc[state, :]
            target_actions = target_actions.reindex(np.random.permutation(target_actions.index))
            target_action = target_actions.idxmax()
        else:
            target_action = np.random.choice(self.actions)
        return target_action

    def update_q_value(self, state, action, reward, state_next):
        self.check_if_state_exist(state_next)
        q_value_predict = self.q_table.loc[state, action]
        if state_next != 'done':
            q_value_real = reward + self.gamma * self.q_table.loc[state_next, :].max()
        else:
            q_value_real = reward
        self.q_table.loc[state, action] += self.alpha * (q_value_real - q_value_predict)

    def check_if_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def train(self):
        for episode in range(100):
            print('Episode: {}'.format(episode))
            state = self.env.reset()
            while True:
                self.env.render()

                # Get next action.
                action = self.get_action(str(state))

                # Get next state.
                state_next, reward, done = self.env.step(action)

                # Update Q table.
                self.update_q_value(str(state), action, reward, str(state_next))

                state = state_next
                if done:
                    break
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    model = QLearning(list(range(env.n_actions)), env)
    env.after(100, model.train)
    env.mainloop()
