"""
Deep Q-Network (DQN) implementation for Self-Driving Chemistry Lab.
Vectorized neural network in pure NumPy with:
- 2-layer MLP (6 -> 64 -> 64 -> 25)
- He (Kaiming) initialization
- Full backward backpropagation
- Adam optimizer with gradient norm clipping
- Experience replay buffer
- Polyak target network soft updating
- Epsilon-greedy exploration schedule
"""

import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int = 15000):
        self.capacity = capacity
        self.states = None
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = None
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        if self.states is None:
            dim = len(state)
            self.states = np.zeros((self.capacity, dim), dtype=np.float32)
            self.next_states = np.zeros((self.capacity, dim), dtype=np.float32)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )

    def __len__(self):
        return self.size


class MLP:
    def __init__(self, in_dim: int, h1: int, h2: int, out_dim: int, seed: int = None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, h1)).astype(np.float32)
        self.b1 = np.zeros((1, h1), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / h1), (h1, h2)).astype(np.float32)
        self.b2 = np.zeros((1, h2), dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / h2), (h2, out_dim)).astype(np.float32)
        self.b3 = np.zeros((1, out_dim), dtype=np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)
        out = np.dot(a2, self.W3) + self.b3
        return out, (x, z1, a1, z2, a2)

    def backward(self, d_out, cache):
        x, z1, a1, z2, a2 = cache
        N = x.shape[0]

        dW3 = np.dot(a2.T, d_out) / N
        db3 = np.sum(d_out, axis=0, keepdims=True) / N

        da2 = np.dot(d_out, self.W3.T)
        dz2 = da2 * (z2 > 0).astype(np.float32)
        dW2 = np.dot(a1.T, dz2) / N
        db2 = np.sum(dz2, axis=0, keepdims=True) / N

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (z1 > 0).astype(np.float32)
        dW1 = np.dot(x.T, dz1) / N
        db1 = np.sum(dz1, axis=0, keepdims=True) / N

        return [dW1, db1, dW2, db2, dW3, db3]

    def copy_weights_from(self, source, tau: float = 1.0):
        for i in range(len(self.params)):
            self.params[i] = tau * source.params[i] + (1.0 - tau) * self.params[i]
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.params


class AdamOptimizer:
    def __init__(self, params, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, max_norm: float = 5.0):
        self.t += 1
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        clip_scale = max_norm / max(total_norm, max_norm)
        for i in range(len(self.params)):
            g = grads[i] * clip_scale
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            self.params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class DQNAgent:
    def __init__(self,
                 state_dim: int = 6,
                 action_dim: int = 25,
                 hidden_dim: int = 64,
                 lr: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.02,
                 epsilon_decay: float = 0.985,
                 buffer_capacity: int = 15000,
                 batch_size: int = 64,
                 tau: float = 0.025,
                 seed: int = 42):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        self.q_net = MLP(state_dim, hidden_dim, hidden_dim, action_dim, seed=seed)
        self.target_net = MLP(state_dim, hidden_dim, hidden_dim, action_dim, seed=seed + 1)
        self.target_net.copy_weights_from(self.q_net, tau=1.0)

        self.optimizer = AdamOptimizer(self.q_net.params, lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_dim))
        
        x = state.reshape(1, -1)
        q_vals, _ = self.q_net.forward(x)
        return int(np.argmax(q_vals[0]))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_curr, cache = self.q_net.forward(states)
        q_target_next, _ = self.target_net.forward(next_states)
        max_q_next = np.max(q_target_next, axis=1)

        targets = rewards + self.gamma * max_q_next * (1.0 - dones)
        batch_idx = np.arange(self.batch_size)
        td_errors = q_curr[batch_idx, actions] - targets
        loss = 0.5 * np.mean(td_errors ** 2)

        d_out = np.zeros_like(q_curr)
        d_out[batch_idx, actions] = td_errors

        grads = self.q_net.backward(d_out, cache)
        self.optimizer.step(grads)

        self.target_net.copy_weights_from(self.q_net, tau=self.tau)
        return float(loss)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
