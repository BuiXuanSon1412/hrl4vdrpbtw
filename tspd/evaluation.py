import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict
import math


def compute_gap(method_cost: float, baseline_cost: float) -> float:
    return (method_cost - baseline_cost) / baseline_cost * 100.0


def evaluate_model(model,env,n_instances: int = 100,n_samples: int = 1,greedy: bool = True,device: str = "cpu",instance_type: str = "random",batch_size: int = 64,) -> Dict:
    model.eval()
    device = torch.device(device)

    from env import generate_tspd_instances

    all_costs = []
    all_times = []

    for start in range(0, n_instances, batch_size):
        B = min(batch_size, n_instances - start)
        coords = generate_tspd_instances(B, env.n_nodes, instance_type, device=str(device))

        t0 = time.time()
        with torch.no_grad():
            if n_samples == 1 or greedy:
                state = env.reset(coords)
                total_reward, _, _ = model.forward(coords, env, state, greedy=True)
                costs = -total_reward.cpu().numpy()
            else:
                # Batch sampling: generate n_samples solutions, take best
                all_sample_rewards = []
                coords_exp = coords.unsqueeze(1).expand(-1, n_samples, -1, -1)
                coords_flat = coords_exp.reshape(B * n_samples, env.n_nodes, 2)

                state = env.reset(coords_flat)
                total_reward, _, _ = model.forward(coords_flat, env, state, greedy=False)
                rewards = total_reward.view(B, n_samples)
                best_rewards = rewards.max(dim=-1).values
                costs = (-best_rewards).cpu().numpy()

        elapsed = time.time() - t0
        all_costs.extend(costs.tolist())
        all_times.append(elapsed / B)  # Per-instance time

    return {
        'mean_cost': np.mean(all_costs),
        'std_cost': np.std(all_costs),
        'mean_time': np.mean(all_times),
        'all_costs': all_costs,
    }


def visualize_solution(coords: torch.Tensor,truck_route: List[int],drone_sorties: List[Tuple[int, int, int]],title: str = "TSP-D Solution",save_path: Optional[str] = None,figsize: Tuple[int, int] = (8, 8),):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    coords_np = coords.numpy() if torch.is_tensor(coords) else coords
    N = len(coords_np)

    # Plot nodes
    depot = coords_np[0]
    customers = coords_np[1:]

    ax.scatter(customers[:, 0], customers[:, 1], c='blue', s=100, zorder=5,
               label='Customer nodes', marker='o')
    ax.scatter(depot[0], depot[1], c='red', s=200, zorder=6,
               label='Depot', marker='s')

    # Annotate nodes
    for i in range(N):
        ax.annotate(str(i), coords_np[i], textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    # Draw truck route (solid lines)
    if truck_route:
        route_coords = coords_np[truck_route]
        for i in range(len(truck_route) - 1):
            ax.annotate(
                "", xy=route_coords[i + 1], xytext=route_coords[i],
                arrowprops=dict(arrowstyle="-|>", color='black', lw=2)
            )

    # Draw drone sorties (dashed lines)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(drone_sorties), 1)))
    for idx, (launch, customer, rendezvous) in enumerate(drone_sorties):
        color = colors[idx % len(colors)]
        # Launch -> customer
        ax.annotate(
            "", xy=coords_np[customer], xytext=coords_np[launch],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            linestyle='dashed', connectionstyle='arc3,rad=0.2')
        )
        # Customer -> rendezvous
        ax.annotate(
            "", xy=coords_np[rendezvous], xytext=coords_np[customer],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            linestyle='dashed', connectionstyle='arc3,rad=0.2')
        )

    # Legend
    truck_patch = mpatches.Patch(color='black', label='Truck path')
    drone_patch = mpatches.Patch(color='green', label='Drone path')
    ax.legend(handles=[truck_patch, drone_patch], loc='upper right')

    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_training_curves(history_ours: Dict,history_hm: Optional[Dict] = None,n_nodes: int = 20,save_path: Optional[str] = None,):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(0, len(history_ours['rewards']) * 200, 200))
    rewards_ours = [-r for r in history_ours['rewards']]  # Convert to costs

    # Learning curve
    ax1 = axes[0]
    ax1.plot(epochs, rewards_ours, 'b-', linewidth=2, label='Ours', alpha=0.8)

    if history_hm:
        rewards_hm = [-r for r in history_hm['rewards']]
        ax1.plot(epochs, rewards_hm, 'r-', linewidth=2, label='HM', alpha=0.8)

    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Average Reward (Lower is Better)')
    ax1.set_title(f'Learning Curves (N={n_nodes})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if history_ours.get('rewards'):
        final_ours = rewards_ours[-1]
        ax1.text(0.98, 0.98,
                 f'Final (Ours): {final_ours:.2f}',
                 transform=ax1.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Cumulative training time
    ax2 = axes[1]
    cum_times_ours = history_ours.get('cumulative_times', [])
    if cum_times_ours:
        ax2.plot(epochs[:len(cum_times_ours)], cum_times_ours,
                 'b-o', linewidth=2, markersize=4, label='Ours', alpha=0.8)

    if history_hm:
        cum_times_hm = history_hm.get('cumulative_times', [])
        if cum_times_hm:
            ax2.plot(epochs[:len(cum_times_hm)], cum_times_hm,
                     'r-o', linewidth=2, markersize=4, label='HM', alpha=0.8)

    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Cumulative Time (s)')
    ax2.set_title(f'Training Time (N={n_nodes})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add avg epoch time annotations
    if history_ours.get('epoch_times'):
        avg_time = np.mean(history_ours['epoch_times'])
        ax2.text(0.02, 0.98,
                 f'Avg epoch: {avg_time:.2f}s',
                 transform=ax2.transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def print_results_table(results: Dict[str, Dict],baseline_key: str = "TSP-ep-all",n_nodes: int = 20,):
    baseline_cost = results[baseline_key]['mean_cost']

    header = f"{'Method':<25} {'Cost':>10} {'Gap':>10} {'Time(s)':>10}"
    print(f"\nTSP-D Results (N={n_nodes})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for method, data in results.items():
        cost = data['mean_cost']
        t = data['mean_time']
        gap = compute_gap(cost, baseline_cost)
        gap_str = f"{gap:+.2f}%"
        marker = " ✓" if method != baseline_key and gap < 0 else ""
        print(f"{method:<25} {cost:>10.3f} {gap_str:>10} {t:>10.3f}{marker}")

    print("=" * len(header))
