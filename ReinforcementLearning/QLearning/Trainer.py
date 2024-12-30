

class Trainer:
    def __init__(self, agent, environment, episodes=1000, max_steps=200):
        self.agent = agent
        self.environment = environment
        self.episodes = episodes
        self.max_steps = max_steps

    def train(self):
        for episode in range(self.episodes):
            state = self.environment.reset()
            total_reward = 0

            for step in range(self.max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.update_q_value(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

            self.agent.decay_epsilon()
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")

    def test(self, episodes=10):
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0

            while True:
                action = self.agent.q_table.best_action(state)
                state, reward, done, _ = self.environment.step(action)
                total_reward += reward
                self.environment.render()

                if done:
                    break

            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")
