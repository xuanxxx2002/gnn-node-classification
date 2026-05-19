import numpy as np
import torch
from torch_geometric.data import DataLoader

from models import GNNStack
from utils.optimizer import build_optimizer


def train(dataset, args):
    print(f"Dataset: {dataset.name} | train nodes: {dataset[0].train_mask.sum().item()}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)
    _, optimizer = build_optimizer(args, model.parameters())

    losses, val_accs = [], []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)[batch.train_mask]
            loss = model.loss(pred, batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            acc = evaluate(loader, model, use_val=True)
            print(f"  Epoch {epoch:>4d}  loss={total_loss:.4f}  val_acc={acc:.4f}")
            val_accs.append(acc)
        else:
            val_accs.append(val_accs[-1] if val_accs else 0.0)

    return val_accs, losses


def evaluate(loader, model, use_val=True):
    model.eval()
    correct = total = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        mask = data.val_mask if use_val else data.test_mask
        correct += pred[mask].eq(data.y[mask]).sum().item()
        total += mask.sum().item()
    return correct / total if total > 0 else 0.0
