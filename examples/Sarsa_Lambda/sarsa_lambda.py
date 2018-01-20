# coding=utf-8

import pandas as pd
import numpy as np

from base.maze import Maze


class SarsaLambda(object):
    def __init__(self, actions, env, alpha=0.01, gamma=0.9, epsilon=0.9, lambda_s=0.9):
        self.actions = actions
        self.env = env
        self.alpha, self.gamma, self.epsilon, self.lambda_s = alpha, gamma, epsilon, lambda_s
        self.q_table = pd.DataFrame(columns=actions, dtype=np.float64)
        self.q_table_trace = pd.DataFrame(columns=actions, dtype=np.float64)

    def check_if_state_exist(self, state):
        if state not in self.q_table.index:

            row = pd.Series(
                    data=[0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
            )

            self.q_table = self.q_table.append(row)
            self.q_table_trace = self.q_table_trace.append(row)

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

        # Get predicted Q value.
        q_value_predict = self.q_table.loc[state, action]

        # Get real Q value.
        if state_next != 'terminal':
            q_value_real = reward + self.gamma * self.q_table.loc[state_next, action_next]
        else:
            q_value_real = reward

        # Calculate error.
        error = q_value_real - q_value_predict

        # Update trace.
        self.q_table_trace.loc[state, :] *= 0
        self.q_table_trace.loc[state, action] = 1

        # Update Q value for current state.
        self.q_table += self.alpha * error * self.q_table_trace

        # Decay trace.
        self.q_table_trace *= self.gamma * self.lambda_s

    def train(self):
        for episode in range(100):

            # Reset state.
            state = self.env.reset()

            # Get first action.
            action = self.get_next_action(str(state))

            self.q_table_trace *= 0

            while True:
                self.env.render()

                # Get next state.
                state_next, reward, done = self.env.step(action)

                # Get next action.
                action_next = self.get_next_action(str(state_next))

                # Update Q table.
                self.update_q_table(str(state), action, reward, str(state_next), action_next)

                state, action = state_next, action_next

                if done:
                    break

            print('For episode: {}, the Q table is:\n {}\n,'
                  ' the Q trace table is:\n {}',
                  episode,
                  self.q_table,
                  self.q_table_trace)


if __name__ == '__main__':
    env = Maze()
    model = SarsaLambda(actions=list(range(env.n_actions)), env=env)
    env.after(50, model.train)
    env.mainloop()
