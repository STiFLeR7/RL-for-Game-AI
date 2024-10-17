# Deep Q-Learning for CartPole Environment

This project implements a Deep Q-Learning agent to solve the CartPole environment using TensorFlow and PyTorch. The agent learns to balance a pole on a cart by taking actions based on the current state of the environment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Agent](#agent)
- [Dependencies](#dependencies)

## Installation

To set up the project, clone this repository and install the required dependencies:

```bash
git clone https://github.com/STiFLeR7/RL-for-Game-AI.git
cd RL-for-Game-AI
pip install -r requirements.txt
```

Make sure to install a compatible version of Gym and its dependencies.

## Usage

1. Ensure you have Python 3.7 or higher installed.
2. Activate your virtual environment (if you're using one).
3. Run the training script:

```bash
python train.py
```

4. The agent will train over a specified number of episodes and save the model as `dqn_model.pth`.

## Training

You can customize the number of training episodes in the `train_agent` function within the `train.py` script. The default is set to 1000 episodes.

## Agent

The agent uses a Deep Q-Network (DQN) approach to approximate the Q-values for each action based on the current state. It employs experience replay and epsilon-greedy strategy for action selection.

### DQN Architecture

- **Input Layer**: Takes the state representation from the environment.
- **Hidden Layers**: Fully connected layers with ReLU activations.
- **Output Layer**: Outputs Q-values for each possible action.

## Dependencies

The following Python packages are required to run this project:

- gym
- numpy
- torch
- tensorflow
- matplotlib

You can install them using pip:

```bash
pip install gym numpy torch tensorflow matplotlib
```
 project or raise issues if you encounter any problems.
