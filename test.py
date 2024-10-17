import gym
import numpy as np
import torch
from dqn_agent import DQNAgent

def test_agent(num_episodes=10):
    env = gym.make('CartPole-v1', render_mode="human")  # Set render mode to "human"
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load('dqn_model.pth', weights_only=True))  # Load the trained model safely

    for episode in range(num_episodes):
        state, _ = env.reset()  # Handle the new reset return structure (state, info)
        done = False
        total_reward = 0

        while not done:
            env.render()  # Render the environment
            state = np.array(state)  # Ensure state is a numpy array
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
            q_values = agent.model(state_tensor)  # Pass the tensor to the model
            action = np.argmax(q_values.detach().numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)  # Handle the new step() structure

            done = terminated or truncated  # Combine terminated and truncated to get "done"
            total_reward += reward
            state = next_state

        print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    test_agent()
