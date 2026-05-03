import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_DATA_DIR = "downsampled"


class EEGCacheDataset(Dataset):
    def __init__(self, data_dir=DEFAULT_DATA_DIR, num_subjects=52, normalize=True):
        self.data_dir = Path(data_dir)
        cache_files = sorted(
            self.data_dir.glob("subject_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )

        self.cache_files = cache_files[:num_subjects]

        print(f"Loading {len(self.cache_files)} subjects from {self.data_dir}...")
        X_list, y_list = [], []

        for i, file_path in enumerate(self.cache_files):
            print(f"\rLoading {file_path.name} ({i + 1}/{len(self.cache_files)})...", end="", flush=True)
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            X_list.append(X_sub)
            y_list.extend(y_sub)

        print(f"\nSuccessfully loaded {len(self.cache_files)} subjects.")

        if not X_list:
            raise ValueError(f"No subject_*.pt files found in {self.data_dir}. Run moabb_data.py first.")

        X_combined = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
        y_combined = np.array(y_list)
        self.classes_, y_encoded = np.unique(y_combined, return_inverse=True)

        if normalize:
            mean = X_combined.mean(axis=-1, keepdims=True)
            std = X_combined.std(axis=-1, keepdims=True)
            X_combined = (X_combined - mean) / np.maximum(std, 1e-7)
            X_combined = np.nan_to_num(X_combined, copy=False)

        self.X_tensor = torch.from_numpy(X_combined).unsqueeze(1)
        self.y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        self.num_classes = len(self.classes_)
        self.n_channels = self.X_tensor.shape[2]
        self.n_timepoints = self.X_tensor.shape[3]

        print(f"Dataset Tensor Shape: {self.X_tensor.shape}")
        print(f"Labels Tensor Shape: {self.y_tensor.shape}")
        print(f"Classes: {', '.join(map(str, self.classes_))}")

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


class AugmentedSubset(Dataset):
    def __init__(
        self,
        dataset,
        indices,
        augment=False,
        noise_std=0.03,
        time_shift=16,
        channel_dropout=0.1,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.augment = augment
        self.noise_std = noise_std
        self.time_shift = time_shift
        self.channel_dropout = channel_dropout

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.augment:
            x = self._augment(x)
        return x, y

    def _augment(self, x):
        x = x.clone()

        if self.time_shift > 0:
            shift = int(torch.randint(-self.time_shift, self.time_shift + 1, (1,)).item())
            if shift:
                x = torch.roll(x, shifts=shift, dims=-1)

        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if self.channel_dropout > 0:
            keep_mask = torch.rand(x.shape[-2], device=x.device) > self.channel_dropout
            if keep_mask.any():
                x = x * keep_mask.view(1, -1, 1)

        return x


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class EEGTransformer(nn.Module):
    def __init__(
        self,
        n_channels,
        n_timepoints,
        n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        temporal_filters=8,
        temporal_kernel_sizes=(15, 31, 63),
        spatial_depth=2,
        dropout=0.4,
    ):
        super().__init__()

        self.temporal_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, temporal_filters, kernel_size=(1, kernel_size), padding=(0, kernel_size // 2), bias=False),
                    nn.BatchNorm2d(temporal_filters),
                    nn.ELU(),
                )
                for kernel_size in temporal_kernel_sizes
            ]
        )

        temporal_channels = temporal_filters * len(temporal_kernel_sizes)
        spatial_channels = temporal_channels * spatial_depth
        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                temporal_channels,
                spatial_channels,
                kernel_size=(n_channels, 1),
                groups=temporal_channels,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_channels),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                spatial_channels,
                spatial_channels,
                kernel_size=(1, 15),
                padding=(0, 7),
                groups=spatial_channels,
                bias=False,
            ),
            nn.Conv2d(spatial_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            SqueezeExcite2d(d_model),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_channels, n_timepoints)
            seq_len = self._extract_features(dummy_input).shape[1]
        if was_training:
            self.train()

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def _extract_features(self, x):
        branch_features = [branch(x) for branch in self.temporal_branches]
        x = torch.cat(branch_features, dim=1)
        x = self.spatial_block(x)
        x = self.separable_block(x)
        return x.squeeze(2).permute(0, 2, 1)

    def forward(self, x):
        x = self._extract_features(x)
        pos_embedding = self.pos_embedding
        if x.shape[1] != pos_embedding.shape[1]:
            pos_embedding = F.interpolate(
                pos_embedding.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        x = x + pos_embedding

        x = self.transformer(x)
        attention_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = (x * attention_weights).sum(dim=1)
        return self.classifier(x)


def stratified_split_indices(labels, train_frac=0.7, val_frac=0.2, seed=42):
    labels = np.asarray(labels)
    generator = np.random.default_rng(seed)
    train_indices, val_indices, test_indices = [], [], []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        generator.shuffle(class_indices)

        train_end = int(train_frac * len(class_indices))
        val_end = train_end + int(val_frac * len(class_indices))
        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:])

    generator.shuffle(train_indices)
    generator.shuffle(val_indices)
    generator.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def create_loaders(
    data_dir=DEFAULT_DATA_DIR,
    num_subjects=52,
    batch_size=32,
    seed=42,
    augment=True,
    noise_std=0.03,
    time_shift=16,
    channel_dropout=0.1,
):
    full_dataset = EEGCacheDataset(data_dir=data_dir, num_subjects=num_subjects)

    train_indices, val_indices, test_indices = stratified_split_indices(
        full_dataset.y_tensor.numpy(),
        train_frac=0.7,
        val_frac=0.2,
        seed=seed,
    )
    train_dataset = AugmentedSubset(
        full_dataset,
        train_indices,
        augment=augment,
        noise_std=noise_std,
        time_shift=time_shift,
        channel_dropout=channel_dropout,
    )
    val_dataset = AugmentedSubset(full_dataset, val_indices, augment=False)
    test_dataset = AugmentedSubset(full_dataset, test_indices, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Split sizes: train={len(train_dataset)} ({len(train_dataset) / len(full_dataset):.0%}), "
        f"val={len(val_dataset)} ({len(val_dataset) / len(full_dataset):.0%}), "
        f"test={len(test_dataset)} ({len(test_dataset) / len(full_dataset):.0%})"
    )

    return full_dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def average_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    return total_loss / total_samples


def soft_cross_entropy(logits, soft_targets, class_weights=None):
    log_probs = F.log_softmax(logits, dim=1)
    if class_weights is not None:
        log_probs = log_probs * class_weights.view(1, -1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def smooth_one_hot(labels, num_classes, smoothing=0.0):
    targets = F.one_hot(labels, num_classes=num_classes).float()
    if smoothing > 0:
        targets = targets * (1.0 - smoothing) + smoothing / num_classes
    return targets


def mixup_batch(inputs, targets, alpha):
    if alpha <= 0:
        return inputs, targets

    lam = float(np.random.beta(alpha, alpha))
    batch_size = inputs.size(0)
    permutation = torch.randperm(batch_size, device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    mixed_targets = lam * targets + (1.0 - lam) * targets[permutation]
    return mixed_inputs, mixed_targets


def train_model(
    full_dataset,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=50,
    lr=0.001,
    weight_decay=1e-3,
    checkpoint_path="best_model.pt",
    label_smoothing=0.05,
    mixup_alpha=0.2,
    grad_clip=1.0,
    save_loss_plot=True,
    loss_plot_path="loss.png",
    return_metrics=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGTransformer(
        n_channels=full_dataset.n_channels,
        n_timepoints=full_dataset.n_timepoints,
        n_classes=full_dataset.num_classes,
    ).to(device)

    class_counts = torch.bincount(full_dataset.y_tensor, minlength=full_dataset.num_classes).float()
    class_weights = (class_counts.sum() / (full_dataset.num_classes * class_counts.clamp_min(1))).to(device)
    print(f"Class counts: {class_counts.int().tolist()}")
    print(f"Class weights: {[round(float(weight), 4) for weight in class_weights.cpu()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = float("-inf")
    best_epoch = 0
    completed_epochs = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            soft_targets = smooth_one_hot(labels, full_dataset.num_classes, smoothing=label_smoothing)
            inputs, soft_targets = mixup_batch(inputs, soft_targets, mixup_alpha)
            outputs = model(inputs)
            loss = soft_cross_entropy(outputs, soft_targets, class_weights=class_weights)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_samples += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(
                    f"\rEpoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"| Current Loss: {loss.item():.4f}",
                    end="",
                )

        scheduler.step()
        train_loss = running_loss / train_samples
        val_loss = average_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print()
        completed_epochs = epoch + 1

        val_acc = accuracy(model, val_loader, device)
        val_accuracies.append((epoch + 1, val_acc))
        current_lr = scheduler.get_last_lr()[0]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% "
            f"| Best: {best_val_acc:.2f}% @ epoch {best_epoch} | LR: {current_lr:.6f}\n"
        )

    if save_loss_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, completed_epochs + 1), train_losses, label="Training Loss", linewidth=2)
        plt.plot(range(1, completed_epochs + 1), val_losses, label="Validation Loss", linewidth=2)
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        print(f"Loss graph saved as {loss_plot_path}")
        plt.close()

    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(
            f"Loaded best validation model from epoch {best_epoch} "
            f"({best_val_acc:.2f}% val accuracy)."
        )

    test_acc = None
    if test_loader is not None:
        test_acc = accuracy(model, test_loader, device)
        print(f"Final Test Accuracy: {test_acc:.2f}%")

    metrics = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "test_acc": test_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }
    if return_metrics:
        return model, metrics
    return model


def run_scaling_law(
    full_dataset,
    train_dataset,
    val_loader,
    data_fractions=None,
    epochs=100,
    batch_size=32,
    output_path="scaling_law.png",
    lr=0.001,
    weight_decay=1e-3,
    label_smoothing=0.05,
    mixup_alpha=0.2,
    grad_clip=1.0,
):
    if data_fractions is None:
        data_fractions = [0.125, 0.25, 0.5, 1.0]

    val_accuracies = []

    for frac in data_fractions:
        print(f"Training on {frac * 100:g}% of the training data...")

        num_samples = int(len(train_dataset) * frac)
        train_subset = Subset(train_dataset, list(range(num_samples)))
        subset_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        checkpoint_path = Path(".dist") / f"scaling_law_{frac:g}.pt"

        _, metrics = train_model(
            full_dataset,
            subset_train_loader,
            val_loader,
            test_loader=None,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            checkpoint_path=checkpoint_path,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            grad_clip=grad_clip,
            save_loss_plot=False,
            return_metrics=True,
        )

        val_acc = metrics["best_val_acc"]
        val_accuracies.append(val_acc)
        print(
            f"Best validation accuracy (evaluated on fixed 20% validation set) "
            f"for {frac * 100:g}% train data: {val_acc:.2f}% "
            f"at epoch {metrics['best_epoch']}\n"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(
        [frac * len(train_dataset) for frac in data_fractions],
        val_accuracies,
        marker="o",
        linestyle="-",
        color="b",
        linewidth=2,
        markersize=8,
    )
    plt.title("Scaling Law: Model Performance vs Dataset Size\n(Evaluated on the same fixed validation set)")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved as {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train EEG transformer on cached MOABB data.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--num-subjects", type=int, default=52)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint-path",
        default="best_model.pt",
        help="Where to save the best validation model weights.",
    )
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

    full_dataset, _, _, _, train_loader, val_loader, test_loader = create_loaders(
        data_dir=args.data_dir,
        num_subjects=args.num_subjects,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_model(
        full_dataset,
        train_loader,
        val_loader,
        test_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
