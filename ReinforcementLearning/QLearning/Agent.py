import numpy as np
from QTable import QTable


class Agent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.q_table = QTable(state_size, action_size)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Explore
        return self.q_table.best_action(state)  # Exploit

    def update_q_value(self, state, action, reward, next_state, done):
        best_next_action = self.q_table.best_action(next_state)
        target = reward + (0 if done else self.gamma * self.q_table.get(next_state, best_next_action))
        new_value = (1 - self.alpha) * self.q_table.get(state, action) + self.alpha * target
        self.q_table.update(state, action, new_value)

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
