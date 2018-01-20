# coding=utf-8

import pandas as pd
import numpy as np

from base.maze import Maze

class Sarsa(object):
    def __init__(self, actions, env, alpha=0.01, gamma=0.9, epsilon=0.9):
        self.actions = actions
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_if_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def get_next_action(self, state):
        self.check_if_state_exist(state)
        if np.random.rand() < self.epsilon:
            target_actions = self.q_table.loc[state, :]
            target_actions = target_actions.reindex(np.random.permutation(target_actions.index))
            target_action = target_actions.idxmax()
        else:
            target_action = np.random.choice(self.actions)
        return target_action

    def update_q_table(self, state, action, reward, state_next, action_next):
        self.check_if_state_exist(state_next)
        q_value_predict = self.q_table.loc[state, action]
        if state_next != 'terminal':
            q_value_real = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_value_real = reward
        self.q_table.loc[state, action] += self.alpha * (q_value_real - q_value_predict)

    def train(self):
        for episode in range(100):

            # Init state.
            state = self.env.reset()

            # Get first action.
            action = self.get_next_action(str(state))

            while True:
                self.env.render()

                # Get next state.
                state_next, reward, terminal = self.env.step(action)

                # Get next action.
                action_next = self.get_next_action(str(state_next))

                # Update Q table.
                self.update_q_table(str(state), action, reward, str(state_next), action_next)

                state, action = state_next, action_next

                if terminal:
                    break

            print('For episode: {}, the Q table is:\n {}'.format(episode, self.q_table))

        print('Game Over')
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    model = Sarsa(actions=list(range(env.n_actions)), env=env)
    env.after(50, model.train)
    env.mainloop()
