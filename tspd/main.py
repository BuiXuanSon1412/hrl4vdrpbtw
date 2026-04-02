import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from actor_critic import TSPDActorCritic
from env import TSPDEnvironment, generate_tspd_instances
from trainer import TSPDTrainer
from evaluation import evaluate_model, plot_training_curves, print_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="TSP-D DRL Training and Evaluation")
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['train', 'eval', 'demo'],
                        help='Run mode: train / eval / demo')
    parser.add_argument('--n_nodes', type=int, default=20,
                        help='Number of nodes (including depot)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples for inference')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--n_encoder_layers', type=int, default=3,
                        help='Number of EGA encoder layers')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--instance_type', type=str, default='random',
                        choices=['random', 'uniform'],
                        help='Instance generation type')
    parser.add_argument('--model_path', type=str, default='tspd_model.pt',
                        help='Path to save/load model')
    parser.add_argument('--drone_speed', type=float, default=2.0,
                        help='Drone speed ratio vs truck (default=2)')
    return parser.parse_args()


def run_demo(args):
    print("TSP-D DRL Demo")
    print("Paper: An End-to-End DRL Approach for Solving TSPD")

    device = args.device
    n_nodes = args.n_nodes

    # Create trainer
    trainer = TSPDTrainer(
        n_nodes=n_nodes,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        n_mgu_layers=2,
        lr=args.lr,
        n_epochs=args.epochs,
        n_cycles=5,
        priority_threshold=0.5,
        validate_every=max(50, args.epochs // 4),
        device=device,
        instance_type=args.instance_type,
        drone_speed_ratio=args.drone_speed,
    )

    # Print model info
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Encoder layers: {args.n_encoder_layers} EGA layers")
    print(f"Decoder: {2} MGU layers")
    print(f"Problem: N={n_nodes} nodes, drone_speed={args.drone_speed}x truck")
    print()

    # Train
    history = trainer.train(verbose=True)

    # Save model
    trainer.save_model(args.model_path)

    # Evaluate with different sampling sizes
    print("Evaluation Results")

    env = TSPDEnvironment(
        n_nodes=n_nodes,
        batch_size=32,
        drone_speed_ratio=args.drone_speed,
        device=device,
    )

    results = {}

    # Greedy evaluation
    res_greedy = evaluate_model(
        trainer.model, env, n_instances=100,
        greedy=True, device=device,
        instance_type=args.instance_type, batch_size=32
    )
    results['Ours(greedy)'] = res_greedy
    print(f"Greedy: Cost={res_greedy['mean_cost']:.3f}, Time={res_greedy['mean_time']:.4f}s")

    # Sampling evaluation
    for k in [100]:
        res_k = evaluate_model(
            trainer.model, env, n_instances=32,
            n_samples=k, greedy=False, device=device,
            instance_type=args.instance_type, batch_size=4
        )
        results[f'Ours(sampling_{k})'] = res_k
        print(f"Sampling_{k}: Cost={res_k['mean_cost']:.3f}, Time={res_k['mean_time']:.4f}s")

    # Plot training curves
    plot_training_curves(history, n_nodes=n_nodes, save_path='training_curves.png')
    print(f"\nTraining curves saved to training_curves.png")

    return history, results


def run_training(args):
    trainer = TSPDTrainer(
        n_nodes=args.n_nodes,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
        lr=args.lr,
        n_epochs=args.epochs,
        n_cycles=5,
        validate_every=500,
        device=args.device,
        instance_type=args.instance_type,
        drone_speed_ratio=args.drone_speed,
    )

    history = trainer.train(verbose=True)
    trainer.save_model(args.model_path)
    return history


def run_evaluation(args):
    device = torch.device(args.device)

    model = TSPDActorCritic(
        n_nodes=args.n_nodes,
        d_model=args.d_model,
        n_encoder_layers=args.n_encoder_layers,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    env = TSPDEnvironment(
        n_nodes=args.n_nodes,
        batch_size=32,
        drone_speed_ratio=args.drone_speed,
        device=args.device,
    )

    results = {}
    for k in [1, 100, 1200, 2400, 4800]:
        greedy = (k == 1)
        label = 'Ours(greedy)' if greedy else f'Ours(sampling_{k})'
        res = evaluate_model(
            model, env, n_instances=100, n_samples=k,
            greedy=greedy, device=args.device,
            instance_type=args.instance_type, batch_size=16
        )
        results[label] = res

    print_results_table(results, baseline_key='Ours(greedy)', n_nodes=args.n_nodes)


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'train':
        run_training(args)
    elif args.mode == 'eval':
        run_evaluation(args)