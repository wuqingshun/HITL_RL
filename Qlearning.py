import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, q_table,actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = q_table

    def choose_action(self, observation):
        observation=str(observation[0])+"_"+str(observation[1])+"_"+str(observation[2])
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        s_=str(s_[0])+"_"+str(s_[1])+"_"+str(s_[2])
        s= str(s[0]) + "_" + str(s[1]) + "_" + str(s[2])
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal


        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        # else:
        #     q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )