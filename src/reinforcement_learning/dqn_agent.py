"""
Deep Q-Network (DQN) Reinforcement Learning agent.
"""
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


if PYTORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """Deep Q-Network."""

        def __init__(self, state_size: int, action_size: int, hidden_layers: list = None):
            """
            Initialize DQN network.

            Args:
                state_size: Size of state space
                action_size: Size of action space
                hidden_layers: List of hidden layer sizes
            """
            super(DQNNetwork, self).__init__()

            if hidden_layers is None:
                hidden_layers = [128, 64]

            layers = []
            prev_size = state_size

            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, action_size))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            """Forward pass."""
            return self.network(x)


    class ReplayMemory:
        """Experience replay memory for DQN."""

        def __init__(self, capacity: int = 10000):
            """
            Initialize replay memory.

            Args:
                capacity: Maximum memory size
            """
            self.memory = deque(maxlen=capacity)

        def push(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool):
            """
            Add experience to memory.

            Args:
                state: Current state
                action: Action taken
                reward: Reward received
                next_state: Next state
                done: Whether episode is done
            """
            self.memory.append((state, action, reward, next_state, done))

        def sample(self, batch_size: int) -> Tuple:
            """
            Sample random batch from memory.

            Args:
                batch_size: Batch size

            Returns:
                Batch of experiences
            """
            batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            return (
                np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32)
            )

        def __len__(self):
            """Get memory size."""
            return len(self.memory)


    class DQNAgent:
        """Deep Q-Learning agent."""

        def __init__(self, state_size: int, action_size: int, config: Optional[Dict] = None):
            """
            Initialize DQN agent.

            Args:
                state_size: Size of state space
                action_size: Size of action space
                config: Configuration dictionary
            """
            self.state_size = state_size
            self.action_size = action_size
            self.config = config or {}

            # Hyperparameters
            self.gamma = self.config.get('gamma', 0.99)
            self.epsilon = self.config.get('epsilon_start', 1.0)
            self.epsilon_end = self.config.get('epsilon_end', 0.01)
            self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
            self.learning_rate = self.config.get('learning_rate', 0.001)
            self.batch_size = self.config.get('batch_size', 64)
            self.target_update = self.config.get('target_update', 10)

            # Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")

            # Networks
            hidden_layers = self.config.get('network', {}).get('hidden_layers', [128, 64])
            self.policy_net = DQNNetwork(state_size, action_size, hidden_layers).to(self.device)
            self.target_net = DQNNetwork(state_size, action_size, hidden_layers).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            # Optimizer
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

            # Replay memory
            memory_size = self.config.get('memory_size', 10000)
            self.memory = ReplayMemory(memory_size)

            # Training stats
            self.training_step = 0

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            """
            Select action using epsilon-greedy policy.

            Args:
                state: Current state
                training: Whether in training mode

            Returns:
                Selected action
            """
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_size)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

        def store_transition(self, state: np.ndarray, action: int, reward: float,
                           next_state: np.ndarray, done: bool):
            """
            Store transition in replay memory.

            Args:
                state: Current state
                action: Action taken
                reward: Reward received
                next_state: Next state
                done: Whether episode is done
            """
            self.memory.push(state, action, reward, next_state, done)

        def train_step(self) -> Optional[float]:
            """
            Perform one training step.

            Returns:
                Loss value or None if not enough samples
            """
            if len(self.memory) < self.batch_size:
                return None

            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Compute current Q values
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Compute next Q values
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # Compute loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update training step
            self.training_step += 1

            # Update target network
            if self.training_step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            return loss.item()

        def train(self, env, episodes: int = 1000, max_steps: int = 500,
                 render: bool = False) -> Dict[str, list]:
            """
            Train the agent.

            Args:
                env: Environment
                episodes: Number of episodes
                max_steps: Maximum steps per episode
                render: Whether to render environment

            Returns:
                Training history
            """
            logger.info(f"Starting training for {episodes} episodes...")

            history = {
                'episode_rewards': [],
                'episode_lengths': [],
                'losses': []
            }

            for episode in range(episodes):
                state, _ = env.reset() if hasattr(env.reset(), '__iter__') and not isinstance(env.reset(), np.ndarray) else (env.reset(), None)
                episode_reward = 0
                episode_losses = []

                for step in range(max_steps):
                    if render:
                        env.render()

                    # Select and perform action
                    action = self.select_action(state)
                    next_state, reward, done, *_ = env.step(action)

                    # Store transition
                    self.store_transition(state, action, reward, next_state, done)

                    # Train
                    loss = self.train_step()
                    if loss is not None:
                        episode_losses.append(loss)

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                # Record history
                history['episode_rewards'].append(episode_reward)
                history['episode_lengths'].append(step + 1)
                if episode_losses:
                    history['losses'].append(np.mean(episode_losses))

                # Log progress
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(history['episode_rewards'][-10:])
                    logger.info(f"Episode {episode + 1}/{episodes} - "
                              f"Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

            logger.info("Training completed!")
            return history

        def save(self, path: str):
            """
            Save agent.

            Args:
                path: Path to save agent
            """
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'training_step': self.training_step,
            }, path)
            logger.info(f"Agent saved to {path}")

        def load(self, path: str):
            """
            Load agent.

            Args:
                path: Path to load agent
            """
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.training_step = checkpoint['training_step']
            logger.info(f"Agent loaded from {path}")

else:
    # Dummy classes if PyTorch not available
    class DQNNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

    class DQNAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")

    class ReplayMemory:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DQN. Install with: pip install torch")
