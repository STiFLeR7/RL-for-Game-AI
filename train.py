import gym
import torch
import numpy as np
from dqn_agent import DQNAgent

def train_agent(num_episodes=1000):
    env = gym.make('CartPole-v1')
    agent = DQNAgent()
    batch_size = 32

    for episode in range(num_episodes):
        state, _ = env.reset()  # Gym v0.26+ returns two values, state and info.
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)

            # Handle new `gym` API which returns 5 values from `step()`
            next_state, reward, terminated, truncated, info = env.step(action)

            # Combine `terminated` and `truncated` into a single `done` flag
            done = terminated or truncated

            # Ensure reward is cast to float (as expected by agent.remember)
            agent.remember(state, action, float(reward), next_state, done)
            total_reward += reward
            state = next_state

        # Perform experience replay and epsilon update
        agent.replay(batch_size)
        agent.update_epsilon()

        # Logging every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model after all episodes
    torch.save(agent.model.state_dict(), 'dqn_model.pth')
    env.close()

if __name__ == "__main__":
    train_agent()
