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

    def update_q_value(self, state: int, action: int, reward: float, next_state: int, done: bool):
        # Get the best action for the next state
        best_action = self.q_table.best_action(next_state)
        
        # Calculate the new Q-value
        current_q = self.q_table.get(state, action)
        next_q = self.q_table.get(next_state, best_action) if not done else 0
        new_value = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q)
        
        # Update the Q-table with the new value
        self.q_table.update(state, action, new_value)

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
