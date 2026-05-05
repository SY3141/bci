import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from moabb_train import DEFAULT_DATA_DIR, create_loaders, train_model


DEFAULT_DATA_FRACTIONS = (0.125, 0.25, 0.5, 1.0)


def parse_data_fractions(value):
    fractions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fraction = float(item)
        if fraction <= 0 or fraction > 1:
            raise ValueError("Each data fraction must be greater than 0 and less than or equal to 1.")
        fractions.append(fraction)

    if not fractions:
        raise ValueError("At least one data fraction is required.")
    return fractions


def run_scaling_law(
    full_dataset,
    train_dataset,
    val_loader,
    data_fractions=DEFAULT_DATA_FRACTIONS,
    epochs=100,
    batch_size=32,
    output_path="scaling_law.png",
    checkpoint_dir=".dist",
    lr=0.001,
    weight_decay=1e-3,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_accuracies = []
    train_sizes = []

    for fraction in data_fractions:
        num_samples = max(1, int(len(train_dataset) * fraction))
        train_sizes.append(num_samples)
        print(f"Training on {fraction * 100:g}% of the training data ({num_samples} samples)...")

        train_subset = Subset(train_dataset, list(range(num_samples)))
        subset_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        checkpoint_path = checkpoint_dir / f"scaling_law_{fraction:g}.pt"

        _, metrics = train_model(
            full_dataset,
            subset_train_loader,
            val_loader,
            test_loader=None,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            save_loss_plot=False,
            return_metrics=True,
        )

        val_acc = metrics["best_val_acc"]
        val_accuracies.append(val_acc)
        print(
            f"Best validation accuracy for {fraction * 100:g}% train data: "
            f"{val_acc:.2f}% at epoch {metrics['best_epoch']}\n"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        train_sizes,
        val_accuracies,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        markersize=8,
    )
    plt.title("Scaling Law: Model Performance vs Dataset Size")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Scaling law graph saved as {output_path}")
    plt.close()

    return {
        "train_sizes": train_sizes,
        "val_accuracies": val_accuracies,
        "data_fractions": list(data_fractions),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a MOABB EEG scaling-law experiment.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--num-subjects", type=int, default=52)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--fractions", default="0.125,0.25,0.5,1.0")
    parser.add_argument("--output-path", default="scaling_law.png")
    parser.add_argument("--checkpoint-dir", default=".dist")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_fractions = parse_data_fractions(args.fractions)
    full_dataset, train_dataset, _, _, _, val_loader, _ = create_loaders(
        data_dir=args.data_dir,
        num_subjects=args.num_subjects,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    run_scaling_law(
        full_dataset=full_dataset,
        train_dataset=train_dataset,
        val_loader=val_loader,
        data_fractions=data_fractions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output_path,
        checkpoint_dir=args.checkpoint_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
