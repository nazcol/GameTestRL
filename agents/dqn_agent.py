import random
import logging
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_qa.config import (
    DQN_LR, DQN_GAMMA, DQN_EPSILON_START, DQN_EPSILON_END,
    DQN_EPSILON_DECAY, DQN_BATCH_SIZE, DQN_MEMORY_SIZE, DQN_TARGET_UPDATE,
    MODEL_DIR,
)
from game_qa.agents.game_env import Game2048Env

logger = logging.getLogger(__name__)


# network                                                              
class DQN(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# replay buffer                                                        
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buf)

# agent                                                                
class DQNAgent:
    """
    QA-oriented DQN: it balances exploitation (to find high-tile states) with
    forced exploration (intentionally tries low-probability actions to
    surface edge-case bugs).
    """

    MODEL_PATH = MODEL_DIR / "dqn.pt"

    def __init__(self, state_dim: int = Game2048Env.STATE_DIM, n_actions: int = Game2048Env.N_ACTIONS):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=DQN_LR)
        self.memory = ReplayBuffer(DQN_MEMORY_SIZE)

        self.epsilon = DQN_EPSILON_START
        self.steps_done = 0
        self._update_counter = 0

        # track which (state, action) -- in which pairs have been visited for coverage
        self._coverage: set[tuple] = set()

    def select_action(self, state: np.ndarray, force_explore: bool = False) -> int:
        """
        epsilon-greedy with coverage bonus.
        `force_explore=True` ignores Q-values and picks least-visited action.
        """
        if force_explore:
            return random.randint(0, self.n_actions - 1)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
        return int(q.argmax(dim=1).item())

    def push(self, state, action, reward, next_state, done):
        key = (tuple(state.round(2)), action)
        self._coverage.add(key)
        self.memory.push(state, action, reward, next_state, float(done))

    def optimize(self) -> Optional[float]:
        if len(self.memory) < DQN_BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(DQN_BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        self.policy_net.train()
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # policy, target
        with torch.no_grad():
            # double DQN: use policy net to select, target net to evaluate
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + DQN_GAMMA * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._update_counter += 1
        if self._update_counter % DQN_TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(DQN_EPSILON_END, self.epsilon * DQN_EPSILON_DECAY)

    @property
    def coverage_count(self) -> int:
        return len(self._coverage)

    def save(self):
        torch.save(self.policy_net.state_dict(), self.MODEL_PATH)
        logger.info("DQN saved → %s", self.MODEL_PATH)

    def load(self):
        if self.MODEL_PATH.exists():
            self.policy_net.load_state_dict(
                torch.load(self.MODEL_PATH, map_location=self.device)
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info("DQN loaded from %s", self.MODEL_PATH)
