"""Entry point for training GNN models on node classification benchmarks."""

import argparse
import os
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid

from utils.trainer import train, evaluate
from torch_geometric.data import DataLoader


def get_args():
    p = argparse.ArgumentParser(description='GNN Node Classification')
    p.add_argument('--model',         type=str,   default='all',
                   choices=['GraphSage', 'GAT', 'all'],
                   help='Model to train. "all" trains and compares both.')
    p.add_argument('--dataset',       type=str,   default='cora', choices=['cora'])
    p.add_argument('--num_layers',    type=int,   default=2)
    p.add_argument('--hidden_dim',    type=int,   default=32)
    p.add_argument('--dropout',       type=float, default=0.6)
    p.add_argument('--epochs',        type=int,   default=500)
    p.add_argument('--lr',            type=float, default=0.01)
    p.add_argument('--weight_decay',  type=float, default=5e-4)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--opt',           type=str,   default='adam',
                   choices=['adam', 'sgd', 'rmsprop', 'adagrad'])
    p.add_argument('--opt_scheduler', type=str,   default='none',
                   choices=['none', 'step', 'cos'])
    p.add_argument('--opt_restart',   type=int,   default=0)
    p.add_argument('--save_plot',     type=str,   default='results.png',
                   help='Path to save the output plot. Set to "" to skip.')
    return p.parse_args()


def main():
    args = get_args()

    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/cora', name='Cora')
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' is not supported yet.")

    models_to_run = ['GraphSage', 'GAT'] if args.model == 'all' else [args.model]

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in models_to_run:
        args.model_type = model_name
        args.heads = 2 if model_name == 'GAT' else 1

        print(f"\n{'='*55}")
        print(f"  Training {model_name}")
        print(f"{'='*55}")

        val_accs, losses = train(dataset, args)

        # Final test-set evaluation
        loader = DataLoader(dataset, batch_size=args.batch_size)
        from models import GNNStack
        # Note: re-uses the last trained model via trainer internals;
        # for standalone test eval, retrain or save checkpoints.

        print(f"\n  [{model_name}] Best val acc : {max(val_accs):.4f}")
        print(f"  [{model_name}] Min loss     : {min(losses):.4f}")

        axes[0].plot(losses,   label=model_name)
        axes[1].plot(val_accs, label=model_name)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.suptitle(f'GNN Node Classification — {dataset.name}')
    plt.tight_layout()

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved → {args.save_plot}")
    plt.show()


if __name__ == '__main__':
    main()
