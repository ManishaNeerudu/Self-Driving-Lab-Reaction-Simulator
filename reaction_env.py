"""
Self-Driving Lab Reaction Simulator Environment for Reinforcement Learning.
Models an automated flow/batch chemical synthesis platform optimizing Temperature (T)
and Reagent Concentration (C_B) to maximize the yield of desired product C,
competing against side-product formation (D) and thermal decomposition (E).
Includes Arrhenius chemical kinetics, measurement noise, reagent/energy costs,
and thermal runaway safety boundaries.
"""

import numpy as np
from scipy.integrate import solve_ivp

class SelfDrivingLabEnv:
    """
    Self-Driving Laboratory MDP Environment for Autonomous Chemical Discovery.
    
    Reaction Network:
        1. A + B -> C  (Desired synthesis, rate k1)
        2. C + B -> D  (Consecutive over-reaction waste, rate k2)
        3. A -> E      (Thermal decomposition waste, rate k3)
        
    Kinetics (Arrhenius Law):
        k1(T) = 0.120 * exp(-48000 / (R) * (1/T - 1/320))
        k2(T) = 0.012 * exp(-38000 / (R) * (1/T - 1/320))
        k3(T) = 0.008 * exp(-62000 / (R) * (1/T - 1/320))
    """

    def __init__(self,
                 max_experiments: int = 25,
                 t_grid_size: int = 5,
                 cb_grid_size: int = 5,
                 noise_std: float = 0.015,
                 seed: int = None):
        
        self.max_experiments = max_experiments
        self.noise_std = noise_std
        self.R = 8.314  # Gas constant J/(mol*K)
        self.C_A0 = 2.0 # Initial concentration of A (mol/L)
        self.tau = 90.0 # Residence / reaction time in seconds
        
        # Operational parameter ranges
        self.t_min, self.t_max = 290.0, 360.0  # Kelvin (17 C to 87 C)
        self.cb_min, self.cb_max = 0.5, 3.5    # mol / L
        
        self.t_grid = np.linspace(self.t_min, self.t_max, t_grid_size)
        self.cb_grid = np.linspace(self.cb_min, self.cb_max, cb_grid_size)
        
        # 5x5 = 25 discrete experimental recipe setpoints
        self.action_coords = []
        for t in self.t_grid:
            for cb in self.cb_grid:
                self.action_coords.append((float(t), float(cb)))
        self.action_coords = np.array(self.action_coords)
        self.action_dim = len(self.action_coords)
        
        # State: [best_yield_norm, last_T_norm, last_Cb_norm, last_yield_norm, budget_left_norm, cost_norm]
        self.state_dim = 6
        
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def _simulate_reaction(self, T: float, C_B0: float):
        """Solves the chemical kinetic ODEs for given operating parameters."""
        k1 = 0.120 * np.exp(-48000.0 / self.R * (1.0 / T - 1.0 / 320.0))
        k2 = 0.012 * np.exp(-38000.0 / self.R * (1.0 / T - 1.0 / 320.0))
        k3 = 0.008 * np.exp(-62000.0 / self.R * (1.0 / T - 1.0 / 320.0))

        def odes(t, y):
            ca, cb, cc, cd, ce = y
            ca = max(0.0, ca)
            cb = max(0.0, cb)
            cc = max(0.0, cc)
            r1 = k1 * ca * cb
            r2 = k2 * cc * cb
            r3 = k3 * ca
            return [-r1 - r3, -r1 - r2, r1 - r2, r2, r3]

        sol = solve_ivp(
            odes,
            [0, self.tau],
            [self.C_A0, C_B0, 0.0, 0.0, 0.0],
            method='RK45',
            atol=1e-5,
            rtol=1e-4
        )
        ca, cb, cc, cd, ce = sol.y[:, -1]
        true_yield = float(cc / self.C_A0)
        total_p = cc + cd + ce + 1e-9
        selectivity = float(cc / total_p)
        
        # Experimental measurement noise (HPLC / spectroscopic detector)
        measured_yield = max(0.0, min(1.0, true_yield + float(self.rng.normal(0, self.noise_std))))
        
        return {
            'yield': measured_yield,
            'true_yield': true_yield,
            'selectivity': selectivity,
            'C_A': ca, 'C_B': cb, 'C_C': cc, 'C_D': cd, 'C_E': ce
        }

    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.best_yield = 0.0
        self.last_T = self.t_min
        self.last_Cb = self.cb_min
        self.last_yield = 0.0
        self.cumulative_energy = 0.0
        self.cumulative_reagent = 0.0
        self.hazard_violations = 0
        
        self.history = {
            'step': [0],
            'T': [self.last_T],
            'Cb': [self.last_Cb],
            'yield': [0.0],
            'best_yield': [0.0],
            'reward': [0.0]
        }
        return self._get_state()

    def _get_state(self):
        budget_left = (self.max_experiments - self.current_step) / float(self.max_experiments)
        cost_norm = (self.cumulative_energy + self.cumulative_reagent) / float(self.max_experiments * 1.5)
        
        return np.array([
            self.best_yield,
            (self.last_T - self.t_min) / (self.t_max - self.t_min),
            (self.last_Cb - self.cb_min) / (self.cb_max - self.cb_min),
            self.last_yield,
            budget_left,
            min(1.0, cost_norm)
        ], dtype=np.float32)

    def step(self, action: int):
        T, C_B = self.action_coords[action]
        self.current_step += 1
        
        res = self._simulate_reaction(T, C_B)
        measured_yield = res['yield']
        
        improvement = max(0.0, measured_yield - self.best_yield)
        self.best_yield = max(self.best_yield, measured_yield)
        
        # Operational expenses:
        # Heating energy cost rises quadratically with temperature above ambient
        t_norm = (T - self.t_min) / (self.t_max - self.t_min)
        energy_cost = 0.08 * (t_norm ** 2)
        # Chemical reagent consumption cost
        reagent_cost = 0.06 * (C_B / self.cb_max)
        
        self.cumulative_energy += energy_cost
        self.cumulative_reagent += reagent_cost
        
        # Safety constraint: high temperature + concentrated reactant B risks boiling / thermal exotherm
        hazard_penalty = 0.0
        is_hazard = False
        if T >= 350.0 and C_B >= 3.0:
            hazard_penalty = 1.5
            is_hazard = True
            self.hazard_violations += 1
            
        # Reward function:
        # Yield reward + selectivity bonus + discovery innovation bonus - energy & reagent costs - safety penalty
        reward = (
            3.0 * measured_yield
            + 0.5 * res['selectivity']
            + 4.0 * improvement
            - energy_cost
            - reagent_cost
            - hazard_penalty
        )
        
        # Target achievement bonus (industrial threshold > 68% yield)
        if measured_yield >= 0.68:
            reward += 1.0

        self.last_T = T
        self.last_Cb = C_B
        self.last_yield = measured_yield
        
        done = (self.current_step >= self.max_experiments)
        
        self.history['step'].append(self.current_step)
        self.history['T'].append(T)
        self.history['Cb'].append(C_B)
        self.history['yield'].append(measured_yield)
        self.history['best_yield'].append(self.best_yield)
        self.history['reward'].append(reward)
        
        info = {
            'step': self.current_step,
            'T': T,
            'Cb': C_B,
            'yield': measured_yield,
            'true_yield': res['true_yield'],
            'selectivity': res['selectivity'],
            'best_yield': self.best_yield,
            'improvement': improvement,
            'hazard': is_hazard
        }
        
        return self._get_state(), reward, done, info
