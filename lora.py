import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from moabb_train import EEGTransformer, accuracy, stratified_split_indices


DATA_DIR = Path("downsampled")
CHECKPOINT_PATH = Path("best_model.pt")
FINETUNE_DIR = Path("finetune")
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.1
SEED = 42
LORA_TARGETS = ("linear1", "linear2", "attention_pool", "classifier")


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


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.base = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        for param in self.base.parameters():
            param.requires_grad = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_output = self.base(x)
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + lora_output * self.scaling

    def merged_linear(self):
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
        ).to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        delta_weight = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data.copy_(self.base.weight.data + delta_weight.data)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def disable_transformer_fastpath():
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subject_numbers(data_dir=DATA_DIR):
    paths = sorted(
        Path(data_dir).glob("subject_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    return [int(path.stem.split("_")[-1]) for path in paths]


def parse_subjects(value, data_dir=DATA_DIR):
    if value.lower() == "all":
        return subject_numbers(data_dir)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def replace_linear_with_lora(module, target_keywords=LORA_TARGETS, prefix=""):
    replaced = []
    for child_name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        should_replace = isinstance(child, nn.Linear) and any(
            keyword in child_prefix for keyword in target_keywords
        )
        if should_replace:
            setattr(module, child_name, LoRALinear(child))
            replaced.append(child_prefix)
        else:
            replaced.extend(replace_linear_with_lora(child, target_keywords, child_prefix))
    return replaced


def merge_lora_layers(module):
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, child_name, child.merged_linear())
        else:
            merge_lora_layers(child)


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint)
    return checkpoint


def load_best_model(model, checkpoint_path=CHECKPOINT_PATH):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find {checkpoint_path}. Run moabb_train.py first.")
    model.load_state_dict(load_state_dict(checkpoint_path))


def create_loaders(dataset, batch_size=BATCH_SIZE, seed=SEED):
    train_indices, val_indices, test_indices = stratified_split_indices(
        dataset.y_tensor.numpy(),
        train_frac=0.6,
        val_frac=0.2,
        seed=seed,
    )
    train_loader = DataLoader(IndexedDataset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(IndexedDataset(dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(IndexedDataset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_lora_model(model, dataset, train_loader, val_loader, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY):
    criterion = nn.CrossEntropyLoss()
    lora_params = [param for name, param in model.named_parameters() if "lora_" in name]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    best_val_accuracy = float("-inf")
    best_model_state = None

    for epoch in range(epochs):
        model.eval()
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.dropout.train()

        running_loss = 0.0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples
        val_accuracy = accuracy(model, val_loader, device)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Subject {dataset.subject} | Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% "
            f"| Best: {best_val_accuracy:.2f}%"
        )

    if best_model_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_model_state.items()})

    return best_val_accuracy


def build_lora_model(dataset, checkpoint_path, device):
    model = EEGTransformer(
        n_channels=dataset.n_channels,
        n_timepoints=dataset.n_timepoints,
        n_classes=dataset.num_classes,
    )
    load_best_model(model, checkpoint_path)
    freeze_model(model)
    replaced_layers = replace_linear_with_lora(model)
    model.to(device)
    return model, replaced_layers


def fine_tune_subject(
    subject,
    data_dir=DATA_DIR,
    checkpoint_path=CHECKPOINT_PATH,
    output_dir=FINETUNE_DIR,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    seed=SEED,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    dataset = SubjectDataset(subject=subject, data_dir=data_dir)
    train_loader, val_loader, test_loader = create_loaders(dataset, batch_size=batch_size, seed=seed)
    model, replaced_layers = build_lora_model(dataset, checkpoint_path, device)

    print(f"\nFine-tuning subject {subject} from {dataset.file_path}")
    print(f"Classes: {', '.join(map(str, dataset.classes_))}")
    print(
        f"Split sizes: train={len(train_loader.dataset)}, "
        f"validation={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
    print(f"LoRA layers: {len(replaced_layers)}")

    best_val_accuracy = train_lora_model(
        model,
        dataset,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )

    merge_lora_layers(model)
    model.eval()
    merged_val_accuracy = accuracy(model, val_loader, device)
    merged_test_accuracy = accuracy(model, test_loader, device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"subject_{subject}_model.pt"
    torch.save(model.state_dict(), output_path)

    print(f"Subject {subject} best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Subject {subject} merged model validation accuracy: {merged_val_accuracy:.2f}%")
    print(f"Subject {subject} merged model test accuracy: {merged_test_accuracy:.2f}%")
    print(f"Saved fine-tuned model to {output_path}")

    return {
        "subject": subject,
        "model_path": output_path,
        "best_val_accuracy": best_val_accuracy,
        "merged_val_accuracy": merged_val_accuracy,
        "merged_test_accuracy": merged_test_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune best_model.pt with LoRA for MOABB subjects.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--checkpoint-path", default=str(CHECKPOINT_PATH))
    parser.add_argument("--output-dir", default=str(FINETUNE_DIR))
    parser.add_argument("--subjects", default="all", help="Use 'all' or a comma-separated list like 1,2,3.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    disable_transformer_fastpath()
    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint_path)
    output_dir = Path(args.output_dir)
    subjects = parse_subjects(args.subjects, data_dir=data_dir)
    if not subjects:
        raise ValueError(f"No subject_*.pt files found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded base model from {checkpoint_path}")
    print(f"Fine-tuning {len(subjects)} subject(s) on {device}")

    results = []
    for subject in subjects:
        results.append(
            fine_tune_subject(
                subject=subject,
                data_dir=data_dir,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                device=device,
            )
        )

    mean_val_accuracy = np.mean([result["merged_val_accuracy"] for result in results])
    mean_test_accuracy = np.mean([result["merged_test_accuracy"] for result in results])
    print(
        f"\nFine-tuned {len(results)} model(s). "
        f"Mean merged validation accuracy: {mean_val_accuracy:.2f}% | "
        f"Mean merged test accuracy: {mean_test_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()
