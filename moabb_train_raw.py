import argparse
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class EEGCacheDataset(Dataset):
    def __init__(self, data_dir="raw", num_subjects=52, cache_subjects=1):
        self.data_dir = Path(data_dir)
        cache_files = sorted(
            self.data_dir.glob("subject_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )

        self.cache_files = cache_files[:num_subjects]
        if not self.cache_files:
            raise ValueError(f"No subject_*.pt files found in {self.data_dir}. Run moabb_data.py first.")

        self.encoder = LabelEncoder()
        self.cache_subjects = max(1, cache_subjects)
        self._subject_cache = OrderedDict()

        print(f"Indexing {len(self.cache_files)} subjects from {self.data_dir}...")
        subject_lengths = []
        raw_labels_by_subject = []
        n_channels, n_timepoints = None, None

        for i, file_path in enumerate(self.cache_files):
            print(f"\rIndexing {file_path.name} ({i + 1}/{len(self.cache_files)})...", end="", flush=True)
            X_sub, y_sub = torch.load(file_path, weights_only=False)

            if X_sub.shape[0] != len(y_sub):
                raise ValueError(f"{file_path} has {X_sub.shape[0]} epochs but {len(y_sub)} labels.")

            if n_channels is None:
                n_channels = X_sub.shape[1]
                n_timepoints = X_sub.shape[2]
            elif X_sub.shape[1:] != (n_channels, n_timepoints):
                raise ValueError(
                    f"{file_path} has shape {X_sub.shape[1:]}, expected {(n_channels, n_timepoints)}."
                )

            subject_lengths.append(len(y_sub))
            raw_labels_by_subject.append(np.array(y_sub))
            del X_sub, y_sub

        print(f"\nSuccessfully indexed {len(self.cache_files)} subjects.")

        self.subject_offsets = np.concatenate(([0], np.cumsum(subject_lengths))).astype(np.int64)
        self.encoder.fit(np.concatenate(raw_labels_by_subject))
        self.labels_by_subject = [
            torch.tensor(self.encoder.transform(labels), dtype=torch.long) for labels in raw_labels_by_subject
        ]
        self.total_epochs = int(self.subject_offsets[-1])
        self.num_classes = len(self.encoder.classes_)
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        print(f"Dataset Shape: ({self.total_epochs}, 1, {self.n_channels}, {self.n_timepoints})")
        print(f"Labels Shape: ({self.total_epochs},)")

    def __len__(self):
        return self.total_epochs

    def __getitem__(self, idx):
        subject_idx, epoch_idx = self._locate_epoch(idx)
        X_sub = self._load_subject(subject_idx)
        return X_sub[epoch_idx].unsqueeze(0), self.labels_by_subject[subject_idx][epoch_idx]

    def subject_index_for(self, idx):
        subject_idx, _ = self._locate_epoch(idx)
        return subject_idx

    def _locate_epoch(self, idx):
        idx = int(idx)
        if idx < 0:
            idx += self.total_epochs
        if idx < 0 or idx >= self.total_epochs:
            raise IndexError(idx)

        subject_idx = int(np.searchsorted(self.subject_offsets, idx, side="right") - 1)
        epoch_idx = idx - int(self.subject_offsets[subject_idx])
        return subject_idx, epoch_idx

    def _load_subject(self, subject_idx):
        if subject_idx in self._subject_cache:
            self._subject_cache.move_to_end(subject_idx)
            return self._subject_cache[subject_idx]

        X_sub, _ = torch.load(self.cache_files[subject_idx], weights_only=False)
        X_sub = torch.as_tensor(X_sub, dtype=torch.float32)
        self._subject_cache[subject_idx] = X_sub
        self._subject_cache.move_to_end(subject_idx)

        while len(self._subject_cache) > self.cache_subjects:
            self._subject_cache.popitem(last=False)

        return X_sub


def _resolve_root_index(dataset, idx):
    while isinstance(dataset, Subset):
        idx = int(dataset.indices[idx])
        dataset = dataset.dataset
    return dataset, int(idx)


class SubjectGroupedBatchSampler:
    def __init__(self, dataset, batch_size, shuffle=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        groups = defaultdict(list)
        for local_idx in range(len(dataset)):
            current_root, root_idx = _resolve_root_index(dataset, local_idx)
            groups[current_root.subject_index_for(root_idx)].append(local_idx)

        self.groups = list(groups.values())

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        batches = []
        for group in self.groups:
            indices = list(group)
            if self.shuffle:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]
            batches.extend(indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size))

        if self.shuffle:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]

        yield from batches

    def __len__(self):
        return sum(math.ceil(len(group) / self.batch_size) for group in self.groups)


class EEGTransformer(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes, d_model=64, n_heads=8, n_layers=4):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12)),
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 15)),
            nn.Dropout(0.5),
        )

        seq_len = n_timepoints // 15
        self.projection = nn.Linear(40, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.3,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.projection(x)

        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x = self.transformer(x)
        return self.classifier(x[:, 0, :])


def create_loaders(data_dir="raw", num_subjects=52, batch_size=32, seed=42, cache_subjects=1, num_workers=0):
    full_dataset = EEGCacheDataset(data_dir=data_dir, num_subjects=num_subjects, cache_subjects=cache_subjects)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=SubjectGroupedBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, seed=seed),
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=SubjectGroupedBatchSampler(val_dataset, batch_size=batch_size, shuffle=False, seed=seed),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=SubjectGroupedBatchSampler(test_dataset, batch_size=batch_size, shuffle=False, seed=seed),
        num_workers=num_workers,
    )

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


def train_model(full_dataset, train_loader, val_loader, test_loader, epochs=50, lr=0.001, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGTransformer(
        n_channels=full_dataset.n_channels,
        n_timepoints=full_dataset.n_timepoints,
        n_classes=full_dataset.num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
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

        if (epoch + 1) % 10 == 0:
            val_acc = accuracy(model, val_loader, device)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}% | LR: {current_lr:.6f}\n"
            )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Training Loss", linewidth=2)
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.savefig("loss.png", dpi=300, bbox_inches="tight")
    print("Loss graph saved as loss.png")
    plt.close()

    test_acc = accuracy(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    return model


def run_scaling_law(
    full_dataset,
    train_dataset,
    val_loader,
    data_fractions=None,
    epochs=100,
    batch_size=32,
    seed=42,
    num_workers=0,
    output_path="scaling_law.png",
):
    if data_fractions is None:
        data_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_accuracies = []

    for frac in data_fractions:
        print(f"Training on {int(frac * 100)}% of the training data...")

        num_samples = int(len(train_dataset) * frac)
        train_subset = Subset(train_dataset, list(range(num_samples)))
        subset_train_loader = DataLoader(
            train_subset,
            batch_sampler=SubjectGroupedBatchSampler(train_subset, batch_size=batch_size, shuffle=True, seed=seed),
            num_workers=num_workers,
        )

        model = EEGTransformer(
            n_channels=full_dataset.n_channels,
            n_timepoints=full_dataset.n_timepoints,
            n_classes=full_dataset.num_classes,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

        for _ in range(epochs):
            model.train()
            for inputs, labels in subset_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        val_acc = accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        print(
            f"Validation Accuracy (Evaluated on fixed 20% validation set) "
            f"for {int(frac * 100)}% train data: {val_acc:.2f}%\n"
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
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train EEG transformer on cached MOABB data.")
    parser.add_argument("--data-dir", default="raw")
    parser.add_argument("--num-subjects", type=int, default=52)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-subjects", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-scaling", action="store_true")
    parser.add_argument("--scaling-epochs", type=int, default=100)
    args = parser.parse_args()

    full_dataset, train_dataset, _, _, train_loader, val_loader, test_loader = create_loaders(
        data_dir=args.data_dir,
        num_subjects=args.num_subjects,
        batch_size=args.batch_size,
        seed=args.seed,
        cache_subjects=args.cache_subjects,
        num_workers=args.num_workers,
    )
    train_model(full_dataset, train_loader, val_loader, test_loader, epochs=args.epochs)

    if not args.skip_scaling:
        run_scaling_law(
            full_dataset,
            train_dataset,
            val_loader,
            epochs=args.scaling_epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
