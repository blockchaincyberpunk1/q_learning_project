import gym
from q_learning.q_learning_agent import QLearningAgent
# Uncomment the next line if you're using a custom environment
# from envs.simple_game_env import SimpleGameEnv

def train_agent(env, agent, episodes, max_steps_per_episode):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print("Training complete!")
    agent.save("q_table.npy")

def main():
    # Initialize the Gym environment
    env = gym.make('FrozenLake-v1', is_slippery=False)
    # Uncomment the next line if you're using a custom environment
    # env = SimpleGameEnv()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Define the grid size (update this based on your actual grid size)
    GRID_SIZE = 5

    # Create our Q-Learning agent
    agent = QLearningAgent(n_states, n_actions, GRID_SIZE)
    

    # Train the agent
    episodes = 1000
    max_steps_per_episode = 100
    train_agent(env, agent, episodes, max_steps_per_episode)

    # Optionally, you can load the agent from a saved state
    # agent.load("q_table.npy")

if __name__ == "__main__":
    main()
