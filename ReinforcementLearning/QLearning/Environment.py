


class Environment:
    def __init__(self, gym_env):
        self.env = gym_env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()
