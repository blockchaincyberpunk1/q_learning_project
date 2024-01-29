import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, grid_size, learning_rate=0.1, gamma=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.grid_size = grid_size
        
    def choose_action(self, state):
        state_index = self._state_to_index(state)  # Convert state to index
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state_index, :])

    def learn(self, state, action, reward, next_state):
        state_index = self._state_to_index(state)  # Convert state to index
        next_state_index = self._state_to_index(next_state)  # Convert next_state to index
        predict = self.q_table[state_index, action]
        target = reward + self.gamma * np.max(self.q_table[next_state_index, :])
        self.q_table[state_index, action] += self.lr * (target - predict)

    def _state_to_index(self, state):
        # Assuming state is a tuple, like (x, y)
        return state[0] * self.grid_size + state[1]
    
    def save(self, file_name):
        np.save(file_name, self.q_table)

    def load(self, file_name):
        self.q_table = np.load(file_name)
