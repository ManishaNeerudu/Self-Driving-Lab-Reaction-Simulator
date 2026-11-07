# Project 7: Self-Driving Lab Simulator for Chemical Yield Optimization
**Unit 5: Real-World Reinforcement Learning Homework Project**

---

## 👥 Team Members & Roll Numbers

| S.No. | Student Name | Roll Number |
| :---: | :--- | :--- | :--- |
| 1 | **[Bhukya Varshitha]** | **[100523729009]** | 
| 2 | **[Bollu Chinmayi]** | **[100523729010]** | 
| 3 | **[Neerudu Manisha]** | **[100523729048]** | 

---

## 📋 Course Deliverables 

- **Deliverable 1**: One-Page Problem Formulation (State, Action, Reward, Reaction Environment)
- **Deliverable 2**: Implementation of standard algorithm (Deep Q-Network with Experience Replay & Polyak Target Updates)
- **Deliverable 3**: Multi-Seed Learning Curves & Comparative Baselines (vs Random Search, OFAT Chemist Heuristic, and Hill-Climbing)
- **Deliverable 4**: 5-Slide Presentation (Motivation, Method, Results, Limitations & Dual-Use Ethics)

---

## 🔬 Deliverable 1: One-Page Problem Formulation

### 1. Scientific Motivation & Problem Statement
In materials science and pharmaceutical synthesis, discovering optimal reaction conditions has traditionally relied on manual, intuitive trial-and-error or univariate sweeps (**One-Factor-at-a-Time / OFAT**). However, chemical reaction networks feature non-linear, competing pathways where increasing temperature accelerates target formation but also triggers destructive over-reaction and thermal decomposition.

**Self-Driving Laboratories (SDLs)** integrate fluidic robotics, continuous-flow synthesis reactors, and online analytics (HPLC/UV-Vis) to optimize reactions autonomously. Here, an RL agent acts as the automated "robot chemist," sequentially selecting experimental recipes (Temperature and Reactant Concentration) across an automated campaign to maximize product yield while minimizing reagent waste, energy costs, and explosion hazards.

### 2. Environment Dynamics (Van de Vusse Reaction Kinetics)
The continuous reactor models three competing non-isothermal reactions in liquid phase over residence time $\tau = 90.0\text{ s}$ ($C_{A,0} = 2.0\text{ M}$):

$$\begin{aligned}
\text{Reaction 1 (Desired Product):} \quad & A + B \xrightarrow{k_1(T)} C \\
\text{Reaction 2 (Over-Reaction Waste):} \quad & C + B \xrightarrow{k_2(T)} D \\
\text{Reaction 3 (Thermal Decomposition):} \quad & A \xrightarrow{k_3(T)} E
\end{aligned}$$

Governed by coupled differential equations:
$$\frac{dC_A}{dt} = -k_1 C_A C_B - k_3 C_A, \quad \frac{dC_B}{dt} = -k_1 C_A C_B - k_2 C_C C_B$$
$$\frac{dC_C}{dt} = k_1 C_A C_B - k_2 C_C C_B, \quad \frac{dC_D}{dt} = k_2 C_C C_B, \quad \frac{dC_E}{dt} = k_3 C_A$$

Rate constants follow Arrhenius' Law ($T_0 = 320\text{ K}$, $R = 8.314\text{ J/(mol}\cdot\text{K)}$):
$$k_1(T) = 0.120 \exp\left(-\frac{48000}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right)\right)$$
$$k_2(T) = 0.012 \exp\left(-\frac{38000}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right)\right)$$
$$k_3(T) = 0.008 \exp\left(-\frac{62000}{R} \left(\frac{1}{T} - \frac{1}{T_0}\right)\right)$$

### 3. Markov Decision Process (MDP) Specification
* **State Space $\mathcal{S} \subset \mathbb{R}^6$**:
  $$s_t = \left[ Y_{\max, t-1}, \quad \frac{T_{t-1} - T_{\min}}{T_{\max} - T_{\min}}, \quad \frac{C_{B, t-1} - C_{B,\min}}{C_{B,\max} - C_{B,\min}}, \quad Y_{t-1}, \quad \frac{H - t}{H}, \quad \text{Cost}_{\text{norm}, t} \right]^\top$$
* **Action Space $\mathcal{A}$**: 25 candidate operating setpoints spanning $T \in [290\text{ K}, 360\text{ K}]$ and $C_B \in [0.5\text{ M}, 3.5\text{ M}]$.
* **Reward Function $\mathcal{R}(s_t, a_t, s_{t+1})$**:
  $$R_t = 3.0 \cdot Y_t + 0.5 \cdot \text{Selectivity}_t + 4.0 \cdot \max(0, Y_t - Y_{\max}) - c_{\text{energy}}(T_t) - c_{\text{reagent}}(C_{B,t}) - P_{\text{hazard}} + R_{\text{target}}$$
  * $P_{\text{hazard}} = 1.5$ if $T \ge 350.0\text{ K}$ and $C_B \ge 3.0\text{ M}$ (thermal runaway boiling/explosion hazard).
  * $R_{\text{target}} = +1.0$ if yield $\ge 68\%$ (commercial threshold).

---

## 💻 Deliverable 2: Implementation Details

- `reaction_env.py`: Solves stiff kinetic ODEs via Runge-Kutta numerical integration, models detector noise, and enforces physical runaway constraints.
- `dqn_agent.py`: Deep Q-Network implemented in pure NumPy featuring a 2-hidden layer MLP ($6 \to 64 \to 64 \to 25$), circular replay buffer ($N=15,000$), Polyak target network soft updating ($\tau = 0.03$), vectorized Adam optimizer with gradient clipping, and $\epsilon$-greedy exploration decay.
- `baselines.py`: Reference baselines including Random Search, One-Factor-at-a-Time (OFAT) Chemist Heuristic, and Greedy Hill-Climbing.
- `train_and_evaluate.py`: End-to-end execution pipeline that trains the agent across seeds, benchmarks baselines, outputs formatted tables directly to console, and produces publication-quality plots.

---

## 📊 Deliverable 3: Evaluation Benchmark Results

Tested over 50 held-out autonomous campaigns (25 experimental runs per campaign):

| Search Policy | Mean Return | Peak Yield Discovered | Campaign Mean Yield | Experiments to $>65\%$ Target | Operational Cost | Hazard Violations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search** | $50.04 \pm 5.42$ | $72.7\%$ | $49.5\%$ | $3.62$ runs | $1.57$ | $50$ hazards |
| **Greedy Hill-Climbing** | $48.96 \pm 7.39$ | $72.9\%$ | $47.9\%$ | $7.30$ runs | $1.37$ | $53$ hazards |
| **OFAT Chemist Heuristic**| $80.24 \pm 1.15$ | $74.5\%$ | $67.0\%$ | $3.00$ runs | $1.53$ | **$0$ hazards** |
| **DQN Agent (Ours)** | **$90.15 \pm 0.52$** | **$74.7\%$** | **$71.7\%$** | **$1.00$ run** | **$1.10$** | **$0$ hazards** |

### Key Takeaways:
1. **Instant Convergence to Global Peak**: DQN reaches $>65\%$ target yield in its first experimental choice ($1.0$ run vs $3.0$ for OFAT and $7.3$ for Hill-Climbing).
2. **Sustained Campaign Efficacy**: Delivers $71.7\%$ average yield across the 25-step campaign, outperforming OFAT ($67.0\%$) and Random ($49.5\%$).
3. **Flawless Safety Record**: Incurs zero thermal runaway violations while reducing operational expenses to $1.10$.

---

## 📑 Deliverable 4: Summary

### Motivation & The Chemical Synthesis Bottleneck
* Edisonian trial-and-error manual experimentation severely delays chemical and materials discovery.
* Complex non-linear reaction networks mean the optimal condition is an interior, non-convex sweet spot in multi-dimensional space.
* Self-Driving Labs use closed-loop RL to steer reaction conditions towards maximum yield with minimal trials and chemical waste.

### Environment & Mathematical Formulation (MDP)
* Competing van de Vusse kinetic ODEs with Arrhenius rate constants: desired product $A+B \to C$ competes with waste $C+B \to D$ and thermal degradation $A \to E$.
* State: Continuous 6-dimensional historical yield, parameters, remaining budget, and cumulative cost.
* Actions: 25 candidate operating setpoints ($T \in [290, 360]\text{ K}$, $C_B \in [0.5, 3.5]\text{ M}$).
* Multi-objective reward balances yield, selectivity, discovery bonuses, energy/reagent costs, and thermal runaway penalties.

### Method: Deep Q-Network (DQN) Architecture
* 2-hidden layer MLP ($6 \to 64 \to 64 \to 25$) with He weight initialization and ReLU activations.
* Experience replay buffer ($15,000$ transitions) and Polyak soft target tracking ($\tau = 0.03$) prevent divergence.
* Vectorized Adam optimizer with gradient norm clipping ($\|\mathbf{g}\| \le 5.0$) minimizing Bellman Temporal Difference error.

### Results & Multi-Baseline Comparisons
* DQN dominates all baselines, achieving $74.7\%$ peak yield and $+90.15$ mean return.
* Reaches the target $>65\%$ yield in $1.0$ experiment, saving $67\%$ of the exploration budget compared to OFAT.
* Operates with zero thermal runaway hazard triggers (compared to 50+ hazards for Random Search).

### Limitations, Safety & Laboratory Ethics
* **Sim-to-Real Gap**: Real flow reactors experience reactor fouling, pump calibration drift, and reagent batch variance.
* **Physical Safety Shielding**: Hard physical interlocks must clip actions outside verified safe temperature/pressure envelopes.
* **Dual-Use Chemical Ethics**: Autonomous synthesis engines require mandatory automated retrosynthesis screening to prevent the synthesis of dangerous or controlled substances.

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install numpy scipy matplotlib

# 2. Run the complete pipeline (trains, benchmarks, prints results & trace, saves figures)
python train_and_evaluate.py
```
