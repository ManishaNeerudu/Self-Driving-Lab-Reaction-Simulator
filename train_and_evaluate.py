"""
Training, Evaluation and Verification Pipeline for Self-Driving Laboratory RL.
Performs multi-seed DQN training, benchmarks against experimental baselines,
prints formatted results and sample traces to the terminal, and saves figures.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reaction_env import SelfDrivingLabEnv
from dqn_agent import DQNAgent
from baselines import RandomSearchAgent, OFATChemistAgent, GreedyHillClimbingAgent

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_agent(agent, num_campaigns: int = 50, seed_offset: int = 5000):
    """Evaluates an agent over held-out experimental campaigns."""
    returns = []
    peak_yields = []
    mean_yields = []
    steps_to_target = []
    costs = []
    hazards = 0

    target_yield = 0.65

    for c in range(num_campaigns):
        env = SelfDrivingLabEnv(seed=seed_offset + c)
        state = env.reset(seed=seed_offset + c)
        if hasattr(agent, 'reset'):
            agent.reset()

        done = False
        campaign_return = 0.0
        step_achieved = env.max_experiments  # Default if never achieved
        campaign_yields = []

        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            campaign_return += reward
            campaign_yields.append(info['yield'])

            if info['yield'] >= target_yield and step_achieved == env.max_experiments:
                step_achieved = info['step']

            state = next_state

        returns.append(campaign_return)
        peak_yields.append(env.best_yield)
        mean_yields.append(float(np.mean(campaign_yields)))
        steps_to_target.append(step_achieved)
        costs.append(env.cumulative_energy + env.cumulative_reagent)
        hazards += env.hazard_violations

    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'mean_peak_yield': float(np.mean(peak_yields)),
        'std_peak_yield': float(np.std(peak_yields)),
        'campaign_mean_yield': float(np.mean(mean_yields)),
        'mean_steps_to_target': float(np.mean(steps_to_target)),
        'mean_cost': float(np.mean(costs)),
        'total_hazards': hazards
    }


def train_dqn(num_seeds: int = 3, num_episodes: int = 220):
    """Trains DQN across multiple seeds to compute statistical confidence bands."""
    all_seed_rewards = []
    best_agent = None
    best_return = -1e9

    print(f"Starting DQN Training ({num_seeds} seeds, {num_episodes} campaigns each)...")
    for s_idx in range(num_seeds):
        seed = 42 + s_idx * 137
        print(f"\n--- Training Seed {s_idx + 1}/{num_seeds} (seed={seed}) ---")
        env = SelfDrivingLabEnv(seed=seed)
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            hidden_dim=64,
            lr=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.03,
            epsilon_decay=0.985,
            buffer_capacity=15000,
            batch_size=64,
            tau=0.03,
            seed=seed
        )

        seed_rewards = []
        for ep in range(num_episodes):
            state = env.reset(seed=seed * 1000 + ep)
            ep_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(state, evaluate=False)
                next_state, reward, done, info = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)
                agent.train_step()
                state = next_state
                ep_reward += reward

            agent.decay_epsilon()
            seed_rewards.append(ep_reward)

            if (ep + 1) % 50 == 0:
                recent_avg = np.mean(seed_rewards[-20:])
                print(f"Campaign {ep + 1:3d}/{num_episodes} | Epsilon: {agent.epsilon:.3f} | Last 20-Avg Return: {recent_avg:5.2f} | Best Yield: {env.best_yield*100:4.1f}%")

        all_seed_rewards.append(seed_rewards)
        eval_stats = evaluate_agent(agent, num_campaigns=30, seed_offset=8000 + s_idx * 100)
        print(f"Seed {s_idx+1} Eval Return: {eval_stats['mean_return']:.2f} | Peak Yield: {eval_stats['mean_peak_yield']*100:.1f}%")

        if eval_stats['mean_return'] > best_return:
            best_return = eval_stats['mean_return']
            best_agent = agent

    return best_agent, np.array(all_seed_rewards)


def plot_learning_curves(all_seed_rewards, baseline_scores, save_path):
    """Figure 1: DQN Learning Curve with 1 SD band vs Baselines."""
    num_episodes = all_seed_rewards.shape[1]
    episodes = np.arange(1, num_episodes + 1)

    window = 15
    def moving_avg(arr, w):
        return np.convolve(arr, np.ones(w)/w, mode='valid')

    smoothed_seeds = np.array([moving_avg(row, window) for row in all_seed_rewards])
    smoothed_episodes = episodes[window - 1:]

    mean_curve = np.mean(smoothed_seeds, axis=0)
    std_curve = np.std(smoothed_seeds, axis=0)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    plt.plot(smoothed_episodes, mean_curve, color='#1f77b4', lw=2.5, label='DQN Autonomous Agent (Mean ± 1 SD)')
    plt.fill_between(smoothed_episodes, mean_curve - std_curve, mean_curve + std_curve, color='#1f77b4', alpha=0.22)

    colors = {'OFAT Chemist Heuristic': '#2ca02c', 'Greedy Hill-Climbing': '#ff7f0e', 'Random Search': '#7f7f7f'}
    styles = {'OFAT Chemist Heuristic': '--', 'Greedy Hill-Climbing': '-.', 'Random Search': ':'}

    for name, stats in baseline_scores.items():
        if name in colors:
            plt.axhline(stats['mean_return'], color=colors[name], linestyle=styles[name], lw=2.0,
                        label=f"{name} (Mean: {stats['mean_return']:.1f})")

    plt.title('Figure 1: Learning Curve for Autonomous Reaction Optimization (Self-Driving Lab)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Training Campaign (Autonomous Laboratory Experiments)', fontsize=11)
    plt.ylabel('Cumulative Campaign Reward', fontsize=11)
    plt.legend(loc='lower right', frameon=True, fontsize=10, shadow=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reaction_landscape_and_trajectories(agents_dict, save_path, eval_seed: int = 999):
    """Figure 2: 2D Chemical Yield Contour Landscape with Overlaid Sampling Trajectories."""
    env = SelfDrivingLabEnv(seed=eval_seed)
    
    t_vals = np.linspace(env.t_min, env.t_max, 40)
    cb_vals = np.linspace(env.cb_min, env.cb_max, 40)
    T_mesh, Cb_mesh = np.meshgrid(t_vals, cb_vals)
    Yield_mesh = np.zeros_like(T_mesh)

    for i in range(len(cb_vals)):
        for j in range(len(t_vals)):
            res = env._simulate_reaction(T_mesh[i, j], Cb_mesh[i, j])
            Yield_mesh[i, j] = res['true_yield'] * 100.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)
    policies = ['Random Search', 'OFAT Chemist Heuristic', 'DQN Agent']

    for idx, name in enumerate(policies):
        ax = axes[idx]
        agent = agents_dict[name]
        
        cp = ax.contourf(T_mesh, Cb_mesh, Yield_mesh, levels=14, cmap='viridis', alpha=0.85)
        cbar = plt.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Target Product Yield (%)', fontsize=9)

        ax.fill_between([350.0, 360.0], 3.0, 3.5, color='red', alpha=0.25, hatch='//', label='Hazard Zone')

        env.reset(seed=eval_seed)
        if hasattr(agent, 'reset'):
            agent.reset()

        state = env.reset(seed=eval_seed)
        done = False
        t_seq, cb_seq = [], []
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            t_seq.append(info['T'])
            cb_seq.append(info['Cb'])

        ax.plot(t_seq, cb_seq, color='white', linestyle='-', lw=1.2, alpha=0.7)
        sc = ax.scatter(t_seq, cb_seq, c=np.arange(len(t_seq)), cmap='cool', edgecolors='black', s=55, zorder=5)
        ax.scatter(t_seq[0], cb_seq[0], color='yellow', marker='*', s=180, edgecolors='black', label='Start', zorder=6)
        ax.scatter(t_seq[-1], cb_seq[-1], color='red', marker='X', s=140, edgecolors='white', label='Final Choice', zorder=6)

        ax.set_title(f"({chr(65+idx)}) {name}\nPeak Discovered: {env.best_yield*100:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xlabel('Temperature T (K)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Concentration Reagent B (mol/L)', fontsize=10)
            ax.legend(loc='lower left', fontsize=8, framealpha=0.85)

    plt.suptitle('Figure 2: Experimental Search Trajectories Over True Chemical Yield Surface', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_discovery_metrics_bar(eval_results, save_path):
    """Figure 3: Comparative Benchmarks for Discovery Efficiency."""
    policies = list(eval_results.keys())
    peak_yields = [eval_results[p]['mean_peak_yield'] * 100 for p in policies]
    steps_to_target = [eval_results[p]['mean_steps_to_target'] for p in policies]
    costs = [eval_results[p]['mean_cost'] for p in policies]

    x = np.arange(len(policies))
    width = 0.26

    fig, ax1 = plt.subplots(figsize=(10, 5.5), dpi=300)

    rects1 = ax1.bar(x - width, peak_yields, width, label='Peak Yield Discovered (%)', color='#2ca02c', alpha=0.88)
    rects2 = ax1.bar(x, steps_to_target, width, label='Experiments to Reach >65% Target', color='#ff7f0e', alpha=0.88)

    ax1.set_ylabel('Yield (%) / Discovery Steps', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, fontsize=10, fontweight='semibold')

    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, costs, width, label='Operational Cost (Energy + Reagents)', color='#1f77b4', alpha=0.88)
    ax2.set_ylabel('Normalized Cost Units', fontsize=11, fontweight='bold', color='#1f77b4')
    ax2.set_ylim(0, max(costs) * 1.4)

    lines = [rects1, rects2, rects3]
    labels = [r.get_label() for r in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=9.5)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#1f77b4')

    plt.title('Figure 3: Discovery Efficiency, Campaign Speed & Operational Cost Across Policies', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def print_formatted_summary(eval_results):
    """Prints a structured evaluation summary table to the console."""
    print("\n" + "=" * 95)
    print("      SELF-DRIVING LABORATORY (PROJECT 7) - FINAL BENCHMARK SUMMARY (50 CAMPAIGNS)")
    print("=" * 95)
    header = f"{'Policy / Agent':<24} | {'Return (Reward)':<16} | {'Peak Yield':<10} | {'Mean Yield':<10} | {'Steps to >65%':<13} | {'Cost':<6} | {'Hazards':<7}"
    print(header)
    print("-" * 95)

    for policy, stats in eval_results.items():
        ret_str = f"{stats['mean_return']:5.2f} ± {stats['std_return']:4.2f}"
        peak_str = f"{stats['mean_peak_yield']*100:5.1f}%"
        mean_y_str = f"{stats['campaign_mean_yield']*100:5.1f}%"
        steps_str = f"{stats['mean_steps_to_target']:4.1f} runs"
        cost_str = f"{stats['mean_cost']:4.2f}"
        haz_str = f"{stats['total_hazards']}"
        print(f"{policy:<24} | {ret_str:<16} | {peak_str:<10} | {mean_y_str:<10} | {steps_str:<13} | {cost_str:<6} | {haz_str:<7}")

    print("=" * 95)


def print_sample_campaign_trace(dqn_agent):
    """Demonstrates a sample 10-step autonomous campaign of the trained DQN agent."""
    print("\n" + "=" * 95)
    print("      LIVE CAMPAIGN DEMONSTRATION: TRAINED DQN AUTONOMOUS AGENT (FIRST 10 EXPERIMENTS)")
    print("=" * 95)
    
    env = SelfDrivingLabEnv(seed=2026)
    state = env.reset(seed=2026)
    
    print(f"{'Step':<5} | {'Agent':<15} | {'Temp (K)':<10} | {'Conc B (M)':<12} | {'Yield (%)':<10} | {'Selectivity':<12} | {'Best Yield':<10}")
    print("-" * 95)
    
    for step in range(1, 11):
        action = dqn_agent.select_action(state, evaluate=True)
        state, reward, done, info = env.step(action)
        print(f"{step:<5} | {'DQN Agent':<15} | {info['T']:<10.1f} | {info['Cb']:<12.2f} | {info['yield']*100:<10.1f} | {info['selectivity']*100:<11.1f}% | {info['best_yield']*100:<10.1f}%")

    print("=" * 95)


def main():
    print("=================================================================")
    print("Project 7: Self-Driving Lab Simulator for Chemical Yield Optimization")
    print("Unit 5 Real-World Reinforcement Learning Homework Project")
    print("=================================================================")

    # 1. Train DQN Agent across seeds
    best_dqn, all_seed_rewards = train_dqn(num_seeds=3, num_episodes=220)

    # 2. Evaluate Baselines
    print("\n--- Evaluating Baselines across 50 held-out campaigns ---")
    agents = {
        'Random Search': RandomSearchAgent(action_dim=25, seed=777),
        'OFAT Chemist Heuristic': OFATChemistAgent(t_grid_size=5, cb_grid_size=5),
        'Greedy Hill-Climbing': GreedyHillClimbingAgent(t_grid_size=5, cb_grid_size=5, seed=888),
        'DQN Agent': best_dqn
    }

    eval_results = {}
    for name, agent in agents.items():
        stats = evaluate_agent(agent, num_campaigns=50, seed_offset=9000)
        eval_results[name] = stats

    # 3. Print Complete Summary Table and Sample Trace to Console
    print_formatted_summary(eval_results)
    print_sample_campaign_trace(best_dqn)

    # 4. Save JSON Metrics
    json_path = os.path.join(RESULTS_DIR, 'evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[Saved Metrics] -> {json_path}")

    # 5. Generate Figures
    fig1_path = os.path.join(RESULTS_DIR, 'learning_curves.png')
    plot_learning_curves(all_seed_rewards, eval_results, fig1_path)

    fig2_path = os.path.join(RESULTS_DIR, 'reaction_landscape_and_trajectories.png')
    plot_reaction_landscape_and_trajectories(agents, fig2_path)

    fig3_path = os.path.join(RESULTS_DIR, 'discovery_metrics_bar.png')
    plot_discovery_metrics_bar(eval_results, fig3_path)

    print("\nAll simulations, baselines, terminal tables, and figures generated successfully!")

if __name__ == '__main__':
    main()
