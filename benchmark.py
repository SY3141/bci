import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from moabb_train import EEGTransformer, accuracy, stratified_split_indices


DATA_DIR = Path("downsampled")
CHECKPOINT_PATH = Path("best_model.pt")
OUTPUT_PATH = Path("benchmark.csv")
BATCH_SIZE = 32
SEED = 42


class SubjectDataset(Dataset):
    def __init__(self, subject, data_dir=DATA_DIR):
        self.subject = subject
        self.file_path = Path(data_dir) / f"subject_{subject}.pt"
        if not self.file_path.exists():
            raise FileNotFoundError(f"Could not find {self.file_path}.")

        X_sub, y_sub = torch.load(self.file_path, weights_only=False)
        if isinstance(X_sub, torch.Tensor):
            X_sub = X_sub.detach().cpu().numpy()

        X_sub = np.asarray(X_sub, dtype=np.float32)
        y_sub = np.asarray(y_sub)
        self.classes_, y_encoded = np.unique(y_sub, return_inverse=True)

        mean = X_sub.mean(axis=-1, keepdims=True)
        std = X_sub.std(axis=-1, keepdims=True)
        X_sub = (X_sub - mean) / np.maximum(std, 1e-7)
        X_sub = np.nan_to_num(X_sub, copy=False)

        self.X_tensor = torch.from_numpy(X_sub).unsqueeze(1)
        self.y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        self.num_classes = len(self.classes_)
        self.n_channels = self.X_tensor.shape[2]
        self.n_timepoints = self.X_tensor.shape[3]

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def load_model(dataset, checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find {checkpoint_path}. Run moabb_train.py first.")

    model = EEGTransformer(
        n_channels=dataset.n_channels,
        n_timepoints=dataset.n_timepoints,
        n_classes=dataset.num_classes,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model.to(device)


def subject_numbers(data_dir=DATA_DIR):
    paths = sorted(
        Path(data_dir).glob("subject_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    return [int(path.stem.split("_")[-1]) for path in paths]


def benchmark_subject(subject, device):
    dataset = SubjectDataset(subject=subject, data_dir=DATA_DIR)
    train_indices, val_indices, _ = stratified_split_indices(
        dataset.y_tensor.numpy(),
        train_frac=0.8,
        val_frac=0.2,
        seed=SEED,
    )
    val_loader = DataLoader(IndexedDataset(dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    model = load_model(dataset, CHECKPOINT_PATH, device=device)
    val_accuracy = accuracy(model, val_loader, device)

    return {
        "subject": subject,
        "file_path": str(dataset.file_path),
        "total_samples": len(dataset),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "classes": "|".join(map(str, dataset.classes_)),
        "validation_accuracy": val_accuracy,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded model: {CHECKPOINT_PATH}")
    print(f"Benchmarking subjects from: {DATA_DIR}")

    results = []
    for subject in subject_numbers(DATA_DIR):
        result = benchmark_subject(subject, device)
        results.append(result)
        print(f"Subject {subject}: {result['validation_accuracy']:.2f}%")

    fieldnames = [
        "subject",
        "file_path",
        "total_samples",
        "train_samples",
        "validation_samples",
        "classes",
        "validation_accuracy",
    ]
    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    accuracies = [result["validation_accuracy"] for result in results]
    print(f"Saved benchmark results to {OUTPUT_PATH}")
    print(f"Mean validation accuracy: {np.mean(accuracies):.2f}%")


if __name__ == "__main__":
    main()
