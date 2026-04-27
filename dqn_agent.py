"""
Deep Q-Network (DQN) Agent
==========================
Standard DQN with experience replay, target network, and epsilon-greedy
exploration.  Input is the 10-D normalised observation from TeleopEnv.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from . import config as cfg
except ImportError:  # pragma: no cover - direct script execution
    import config as cfg


# ------------------------------------------------------------------ #
#  Replay buffer (pre-allocated numpy circular buffer)                #
# ------------------------------------------------------------------ #

class ReplayBuffer:
    """Fixed-size circular replay buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.rng = np.random.default_rng(seed)

        self.obs      = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions  = np.zeros(self.capacity, dtype=np.int64)
        self.rewards  = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones    = np.zeros(self.capacity, dtype=np.float32)

        self.pos = 0
        self.size = 0

    def push(self, obs, action: int, reward: float, next_obs, done: bool) -> None:
        idx = self.pos
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size


# ------------------------------------------------------------------ #
#  Q-network                                                          #
# ------------------------------------------------------------------ #

class DQNNetwork(nn.Module):
    """Fully connected Q-network with configurable hidden layers."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
#  DQN Agent                                                          #
# ------------------------------------------------------------------ #

class DQNAgent:
    """Standard DQN with target network and experience replay."""

    @staticmethod
    def decay_for_horizon(
        epsilon_start: float,
        epsilon_end: float,
        total_episodes: int,
    ) -> float:
        horizon = max(1, int(total_episodes))
        start = max(float(epsilon_start), 1e-12)
        end = max(min(float(epsilon_end), start), 1e-12)
        return float((end / start) ** (1.0 / horizon))

    def __init__(
        self,
        obs_dim: int = 10,
        n_actions: int = cfg.N_ACTIONS,
        seed: int | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay: float | None = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Hyper-parameters (from config)
        self.gamma      = cfg.DQN_DISCOUNT_FACTOR
        self.batch_size = cfg.DQN_BATCH_SIZE
        self.target_update_freq = cfg.DQN_TARGET_UPDATE_FREQ
        self.epsilon = float(cfg.DQN_EPSILON_START if epsilon_start is None else epsilon_start)
        self.epsilon_end = float(cfg.DQN_EPSILON_END if epsilon_end is None else epsilon_end)
        self.epsilon_decay = float(
            self.decay_for_horizon(self.epsilon, self.epsilon_end, cfg.DQN_EPSILON_DECAY_EPISODES)
            if epsilon_decay is None
            else epsilon_decay
        )
        self.min_replay_size = cfg.DQN_MIN_REPLAY_SIZE
        self.grad_clip = cfg.DQN_GRAD_CLIP

        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        hidden = cfg.DQN_HIDDEN_SIZES
        self.policy_net = DQNNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = DQNNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=cfg.DQN_LEARNING_RATE)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            cfg.DQN_REPLAY_BUFFER_SIZE, obs_dim, seed=seed,
        )
        self.train_step_count = 0

    # ------------------------------------------------------------------ #
    #  Action selection                                                    #
    # ------------------------------------------------------------------ #
    def select_action(self, obs: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            return int(self.policy_net(t).argmax(dim=1).item())

    def q_values_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Return Q(s, :) for a batch of observations without exploration."""
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        with torch.no_grad():
            t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            q = self.policy_net(t).cpu().numpy()
        return np.asarray(q, dtype=np.float64)

    def select_actions_batch(
        self,
        obs_batch: np.ndarray,
        q_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized epsilon-greedy action selection for a batch."""
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        batch_size = int(obs_batch.shape[0])
        if q_values is None:
            q_values = self.q_values_batch(obs_batch)
        q_values = np.asarray(q_values, dtype=np.float64)
        greedy = np.argmax(q_values, axis=1).astype(np.int64, copy=False)
        explore_mask = self.rng.random(batch_size) < self.epsilon
        if np.any(explore_mask):
            greedy = np.array(greedy, copy=True)
            greedy[explore_mask] = self.rng.integers(self.n_actions, size=int(np.sum(explore_mask)))
        return greedy

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q(s, :) for one observation without applying epsilon-greedy."""
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            q = self.policy_net(t).squeeze(0).cpu().numpy()
        return np.asarray(q, dtype=np.float64)

    def greedy_action(self, obs: np.ndarray) -> int:
        """Return argmax_a Q(s, a) without exploration."""
        return int(np.argmax(self.q_values(obs)))

    # ------------------------------------------------------------------ #
    #  Store + learn                                                       #
    # ------------------------------------------------------------------ #
    def store_transition(self, obs, action: int, reward: float,
                         next_obs, done: bool) -> None:
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def train_step(self) -> float | None:
        """One gradient step.  Returns loss or None if buffer too small."""
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b = self.replay_buffer.sample(
            self.batch_size)

        obs_t  = torch.as_tensor(obs_b,  dtype=torch.float32, device=self.device)
        act_t  = torch.as_tensor(act_b,  dtype=torch.int64,   device=self.device)
        rew_t  = torch.as_tensor(rew_b,  dtype=torch.float32, device=self.device)
        nobs_t = torch.as_tensor(nobs_b, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done_b, dtype=torch.float32, device=self.device)

        q_selected = (self.policy_net(obs_t)
                      .gather(1, act_t.unsqueeze(1)).squeeze(1))

        with torch.no_grad():
            next_q = self.target_net(nobs_t).max(dim=1).values
            target = rew_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.functional.mse_loss(q_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                     self.grad_clip)
        self.optimizer.step()
        self.train_step_count += 1

        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------ #
    #  Epsilon decay                                                       #
    # ------------------------------------------------------------------ #
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end,
                           self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "train_step_count": self.train_step_count,
            "obs_dim":    self.obs_dim,
            "n_actions":  self.n_actions,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.train_step_count = ckpt.get("train_step_count", 0)

    def __repr__(self) -> str:
        return (
            f"DQNAgent(obs={self.obs_dim}, actions={self.n_actions}, "
            f"buffer={len(self.replay_buffer)}/{self.replay_buffer.capacity}, "
            f"grad_steps={self.train_step_count}, eps={self.epsilon:.4f})"
        )
