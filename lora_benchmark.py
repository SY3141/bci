import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lora import BATCH_SIZE, DATA_DIR, FINETUNE_DIR, SEED, IndexedDataset, SubjectDataset, disable_transformer_fastpath
from moabb_train import EEGTransformer, accuracy, stratified_split_indices


OUTPUT_PATH = Path("lora_benchmark.csv")


def model_paths(finetune_dir=FINETUNE_DIR):
    return sorted(
        Path(finetune_dir).glob("subject_*_model.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )


def subject_from_model_path(model_path):
    return int(model_path.stem.split("_")[1])


def load_finetuned_model(dataset, model_path, device):
    model = EEGTransformer(
        n_channels=dataset.n_channels,
        n_timepoints=dataset.n_timepoints,
        n_classes=dataset.num_classes,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    return model.to(device)


def benchmark_model(model_path, data_dir, device):
    subject = subject_from_model_path(model_path)
    dataset = SubjectDataset(subject=subject, data_dir=data_dir)
    train_indices, val_indices, test_indices = stratified_split_indices(
        dataset.y_tensor.numpy(),
        train_frac=0.6,
        val_frac=0.2,
        seed=SEED,
    )
    test_loader = DataLoader(IndexedDataset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False)
    model = load_finetuned_model(dataset, model_path, device)
    test_accuracy = accuracy(model, test_loader, device)

    return {
        "subject": subject,
        "model_path": str(model_path),
        "file_path": str(dataset.file_path),
        "total_samples": len(dataset),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "classes": "|".join(map(str, dataset.classes_)),
        "test_accuracy": test_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LoRA fine-tuned MOABB subject models.")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--finetune-dir", default=str(FINETUNE_DIR))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    disable_transformer_fastpath()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_dir = Path(args.data_dir)
    finetune_dir = Path(args.finetune_dir)
    output_path = Path(args.output_path)
    paths = model_paths(finetune_dir)
    if not paths:
        raise FileNotFoundError(f"No subject_*_model.pt files found in {finetune_dir}. Run lora.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking {len(paths)} LoRA fine-tuned model(s) from {finetune_dir}")

    results = []
    for model_path in paths:
        result = benchmark_model(model_path, data_dir, device)
        results.append(result)
        print(f"Subject {result['subject']}: test={result['test_accuracy']:.2f}%")

    fieldnames = [
        "subject",
        "model_path",
        "file_path",
        "total_samples",
        "train_samples",
        "validation_samples",
        "test_samples",
        "classes",
        "test_accuracy",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    test_accuracies = [result["test_accuracy"] for result in results]
    print(f"Saved LoRA benchmark results to {output_path}")
    print(f"Mean LoRA test accuracy: {np.mean(test_accuracies):.2f}%")


if __name__ == "__main__":
    main()
