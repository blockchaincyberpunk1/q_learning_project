import numpy as np
import gym
from gym import spaces

class SimpleGameEnv(gym.Env):
    """
    Simple Grid Environment for Q-Learning
    The agent starts at the top-left corner and needs to reach the bottom-right corner avoiding obstacles.
    """
    def __init__(self, grid_size=5, obstacles=None):
        super(SimpleGameEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        self.grid_size = grid_size
        self.goal_position = (grid_size - 1, grid_size - 1)

        if obstacles is None:
            obstacles = [(1, 1), (2, 2), (3, 3)]  # Default obstacles
        self.obstacles = set(obstacles)

        self.state = None

    def reset(self):
        """
        Resets the environment to the initial state and returns the initial observation.
        """
        self.state = (0, 0)  # Start at the top-left corner
        return self._state_to_observation(self.state)

    def step(self, action):
        """
        Executes the action in the environment.
        Returns next state, reward, done, and additional info.
        """
        x, y = self.state

        # Define the action effects
        if action == 0:  # Up
            y = max(y - 1, 0)
        elif action == 1:  # Down
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:  # Left
            x = max(x - 1, 0)
        elif action == 3:  # Right
            x = min(x + 1, self.grid_size - 1)

        self.state = (x, y)

        # Check if the agent has reached the goal
        done = self.state == self.goal_position

        # Define the reward
        reward = 1 if done else -1 if self.state in self.obstacles else 0

        return self._state_to_observation(self.state), reward, done, {}

    def _state_to_observation(self, state):
        """
        Converts the state to an observation.
        """
        x, y = state
        return y * self.grid_size + x

    def render(self, mode='human'):
        """
        Renders the current state of the environment.
        """
        grid = np.zeros((self.grid_size, self.grid_size))
        for obs in self.obstacles:
            grid[obs] = -1
        grid[self.state] = 1
        grid[self.goal_position] = 0.5
        print(grid)

