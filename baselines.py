"""
Baseline agents for the Self-Driving Laboratory simulator:
1. RandomSearchAgent: Uniform random exploration (materials discovery baseline).
2. OFATChemistAgent: One-Factor-at-a-Time univariate sweep (traditional chemist heuristic).
3. GreedyHillClimbingAgent: Local gradient-ascent hill climbing.
"""

import numpy as np

class RandomSearchAgent:
    """Randomly selects experimental recipes."""
    def __init__(self, action_dim: int = 25, seed: int = None):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> int:
        return int(self.rng.integers(0, self.action_dim))

    def reset(self):
        pass


class OFATChemistAgent:
    """
    One-Factor-at-a-Time (OFAT) Traditional Chemist Protocol:
    1. Fix temperature T at median level, sweep all concentrations C_B (5 trials).
    2. Identify best C_B, hold it fixed, and sweep all temperatures T (5 trials).
    3. Exploit the best discovered combination (T*, C_B*) for the remainder of the campaign.
    """
    def __init__(self, t_grid_size: int = 5, cb_grid_size: int = 5):
        self.t_grid_size = t_grid_size
        self.cb_grid_size = cb_grid_size
        self.reset()

    def reset(self):
        self.step_count = 0
        self.mid_t_idx = self.t_grid_size // 2
        self.best_cb_idx = 0
        self.best_cb_yield = -1.0
        self.best_t_idx = 0
        self.best_t_yield = -1.0
        self.best_action = 0

    def _coord_to_action(self, t_idx: int, cb_idx: int) -> int:
        return t_idx * self.cb_grid_size + cb_idx

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> int:
        # state: [best_yield, last_T_norm, last_Cb_norm, last_yield, budget_left, cost_norm]
        last_yield = state[3]
        
        # Update records from previous step result
        if self.step_count > 0:
            if self.step_count <= self.cb_grid_size:
                # Evaluating C_B sweep
                prev_cb_idx = self.step_count - 1
                if last_yield > self.best_cb_yield:
                    self.best_cb_yield = last_yield
                    self.best_cb_idx = prev_cb_idx
            elif self.step_count <= self.cb_grid_size + self.t_grid_size:
                # Evaluating T sweep
                prev_t_idx = (self.step_count - 1) - self.cb_grid_size
                if last_yield > self.best_t_yield:
                    self.best_t_yield = last_yield
                    self.best_t_idx = prev_t_idx

        # Phase 1: Sweep C_B holding T at median
        if self.step_count < self.cb_grid_size:
            cb_idx = self.step_count
            action = self._coord_to_action(self.mid_t_idx, cb_idx)
            self.step_count += 1
            return action
        
        # Phase 2: Sweep T holding C_B at best from Phase 1
        elif self.step_count < self.cb_grid_size + self.t_grid_size:
            t_idx = self.step_count - self.cb_grid_size
            action = self._coord_to_action(t_idx, self.best_cb_idx)
            self.step_count += 1
            return action
        
        # Phase 3: Exploit optimum
        else:
            action = self._coord_to_action(self.best_t_idx, self.best_cb_idx)
            self.step_count += 1
            return action


class GreedyHillClimbingAgent:
    """
    Local hill-climbing search with exploratory random walks.
    Tests adjacent cells in grid and steps towards highest yield.
    """
    def __init__(self, t_grid_size: int = 5, cb_grid_size: int = 5, seed: int = None):
        self.t_grid_size = t_grid_size
        self.cb_grid_size = cb_grid_size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.curr_t = 0
        self.curr_cb = 0
        self.best_yield = -1.0
        self.stagnant_steps = 0

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> int:
        last_yield = state[3]
        if last_yield > self.best_yield:
            self.best_yield = last_yield
            self.stagnant_steps = 0
        else:
            self.stagnant_steps += 1

        # If stuck, jump to random location
        if self.stagnant_steps >= 3:
            self.curr_t = int(self.rng.integers(0, self.t_grid_size))
            self.curr_cb = int(self.rng.integers(0, self.cb_grid_size))
            self.stagnant_steps = 0
        else:
            # Propose neighbor step
            dt = self.rng.choice([-1, 0, 1])
            dcb = self.rng.choice([-1, 0, 1])
            self.curr_t = int(np.clip(self.curr_t + dt, 0, self.t_grid_size - 1))
            self.curr_cb = int(np.clip(self.curr_cb + dcb, 0, self.cb_grid_size - 1))

        return self.curr_t * self.cb_grid_size + self.curr_cb
