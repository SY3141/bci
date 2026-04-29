import argparse
from pathlib import Path

import torch


INPUT_DATA_DIR = "pygedai_processed"
OUTPUT_DATA_DIR = "spectrogram_data"
DEFAULT_SFREQ = 512 / 5


def get_cache_files(data_dir):
    return sorted(
        Path(data_dir).glob("subject_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )


def make_spectrogram_tensor(
    X_sub,
    sfreq=DEFAULT_SFREQ,
    n_fft=64,
    hop_length=4,
    fmin=8.0,
    fmax=32.0,
    eps=1e-8,
):
    """Convert EEG epochs from (epochs, channels, timepoints) to spectrogram features.

    Output shape is (epochs, channels * frequency_bins, spectrogram_timepoints),
    which keeps the data compatible with moabb_train.py.
    """
    X_sub = torch.as_tensor(X_sub, dtype=torch.float32)
    if X_sub.ndim != 3:
        raise ValueError(f"Expected X_sub with shape (epochs, channels, timepoints), got {tuple(X_sub.shape)}")

    n_epochs, n_channels, n_timepoints = X_sub.shape
    if n_timepoints < n_fft:
        raise ValueError(f"n_fft={n_fft} is larger than the input time dimension {n_timepoints}.")

    flattened = X_sub.reshape(n_epochs * n_channels, n_timepoints)
    window = torch.hann_window(n_fft, dtype=torch.float32)

    stft = torch.stft(
        flattened,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )

    power = stft.abs().pow(2)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if not bool(freq_mask.any()):
        raise ValueError(f"No STFT frequency bins found between {fmin} and {fmax} Hz.")

    log_power = torch.log(power[:, freq_mask, :] + eps)
    n_freqs = log_power.shape[1]
    n_frames = log_power.shape[2]

    spectrogram = log_power.reshape(n_epochs, n_channels, n_freqs, n_frames)
    spectrogram = spectrogram.reshape(n_epochs, n_channels * n_freqs, n_frames)

    mean = spectrogram.mean(dim=-1, keepdim=True)
    std = spectrogram.std(dim=-1, keepdim=True).clamp_min(eps)
    return (spectrogram - mean) / std


def convert_subject_file(
    input_path,
    output_dir=OUTPUT_DATA_DIR,
    sfreq=DEFAULT_SFREQ,
    n_fft=64,
    hop_length=4,
    fmin=8.0,
    fmax=32.0,
):
    subject_num = int(input_path.stem.split("_")[-1])
    X_sub, y_sub = torch.load(input_path, weights_only=False)
    X_spec = make_spectrogram_tensor(
        X_sub,
        sfreq=sfreq,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
    )

    output_path = Path(output_dir) / input_path.name
    torch.save((X_spec, y_sub), output_path)
    print(f"Subject {subject_num}: {tuple(torch.as_tensor(X_sub).shape)} -> {tuple(X_spec.shape)} saved to {output_path}")


def convert_all_subjects(
    input_dir=INPUT_DATA_DIR,
    output_dir=OUTPUT_DATA_DIR,
    sfreq=DEFAULT_SFREQ,
    n_fft=64,
    hop_length=4,
    fmin=8.0,
    fmax=32.0,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_files = get_cache_files(input_path)
    if not cache_files:
        raise ValueError(f"No subject_*.pt files found in {input_path}. Run moabb_data.py first.")

    print(f"Converting {len(cache_files)} subjects from {input_path} to spectrograms in {output_path}...")
    print(f"STFT settings: sfreq={sfreq}, n_fft={n_fft}, hop_length={hop_length}, frequency band={fmin}-{fmax} Hz")

    for file_path in cache_files:
        convert_subject_file(
            file_path,
            output_dir=output_path,
            sfreq=sfreq,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
        )

    print("Spectrogram conversion complete.")


def main():
    parser = argparse.ArgumentParser(description="Convert PyGEDAI-cleaned EEG tensors to spectrogram tensors.")
    parser.add_argument("--input-dir", default=INPUT_DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DATA_DIR)
    parser.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ)
    parser.add_argument("--n-fft", type=int, default=64)
    parser.add_argument("--hop-length", type=int, default=4)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=32.0)
    args = parser.parse_args()

    convert_all_subjects(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sfreq=args.sfreq,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
    )


if __name__ == "__main__":
    main()
