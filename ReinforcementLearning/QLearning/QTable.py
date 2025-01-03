import numpy as np


class QTable:
    def __init__(self, state_size, action_size, initial_value=0.0):
        self.q_table = np.full((state_size, action_size), initial_value)

    def get(self, state:int, action:int):
        return self.q_table[state, action]

    def update(self, state:int, action:int, value):
        self.q_table[state, action] = value

    def best_action(self, state:int):
        return np.argmax(self.q_table[state])
