import gym
from gym import spaces

class DroneEnvironment(gym.Env):
    def __init__(self):
        self.state_dim = 4  # define state dimension (4 rotors)
        self.action_dim = 4  # define action dimension
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,))
        self.action_space = spaces.Discrete(self.action_dim)

    def reset(self):
        # reset environment
        pass

    def step(self, action):
        # take action in environment
        pass
