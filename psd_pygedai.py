from pathlib import Path
import gc

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch


DATA_DIR = Path("pygedai_processed")
OUTPUT_DIR = Path("pygedai_processed_psd")
SAMPLING_RATE = 512
MAX_FREQUENCY = 80


def subject_number(subject_path):
    return int(subject_path.stem.split("_")[-1])


def get_subject_files(data_dir=DATA_DIR):
    return sorted(data_dir.glob("subject_*.pt"), key=subject_number)


def load_subject(subject_path):
    data = torch.load(subject_path, weights_only=False)
    if not isinstance(data, tuple) or len(data) < 1:
        raise ValueError(f"Expected {subject_path} to contain a tuple with EEG data.")

    X_sub = data[0]
    if isinstance(X_sub, torch.Tensor):
        X_sub = X_sub.detach().cpu().numpy()

    X_sub = np.asarray(X_sub, dtype=np.float32)
    if X_sub.ndim != 3:
        raise ValueError(
            f"Expected EEG data shaped as (epochs, channels, timepoints), got {X_sub.shape}."
        )

    return X_sub


def compute_average_psd(X_sub, sampling_rate=SAMPLING_RATE):
    n_timepoints = X_sub.shape[-1]
    nperseg = min(1024, n_timepoints)
    frequencies, psd = welch(
        X_sub,
        fs=sampling_rate,
        nperseg=nperseg,
        axis=-1,
        detrend="constant",
        scaling="density",
    )
    mean_psd = psd.mean(axis=(0, 1))
    return frequencies, mean_psd


def plot_psd(frequencies, mean_psd, output_path, subject_label, max_frequency=MAX_FREQUENCY):
    mask = frequencies <= max_frequency

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(frequencies[mask], mean_psd[mask], color="#16a34a", linewidth=2)
    ax.set_title(f"{subject_label} PyGEDAI Average Power Spectral Density")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(0, max_frequency)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find {DATA_DIR}")

    subject_files = get_subject_files(DATA_DIR)
    if not subject_files:
        raise FileNotFoundError(f"No subject_*.pt files found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, subject_path in enumerate(subject_files, start=1):
        subject_label = subject_path.stem.replace("_", " ").title()
        output_path = OUTPUT_DIR / f"{subject_path.stem}_psd.png"
        print(f"[{index}/{len(subject_files)}] Plotting {subject_path} -> {output_path}")

        X_sub = load_subject(subject_path)
        frequencies, mean_psd = compute_average_psd(X_sub)
        plot_psd(frequencies, mean_psd, output_path, subject_label)

        del X_sub, frequencies, mean_psd
        gc.collect()

    print(f"Saved {len(subject_files)} PSD plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
