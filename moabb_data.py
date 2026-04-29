import argparse
import gc
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
import urllib3


PROCESSED_DATA_DIR = "processed_data"
PYGEDAI_DATA_DIR = "pygedai_processed"
MNE_DATA_DIR = "mne_data"
CHO2017_SFREQ = 512


warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)


def _format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024


class NotebookDownloadProgress:
    def __init__(self, total=0, unit="B", unit_scale=True, **kwargs):
        self.total = total or 0
        self.n = 0
        self.unit = unit
        self.unit_scale = unit_scale
        self.started_at = time.time()
        self.last_render_at = 0.0
        self.last_pct = -1
        self._render(force=True)

    def _render(self, force=False):
        now = time.time()
        if not force and now - self.last_render_at < 0.2:
            return
        self.last_render_at = now
        total = max(int(self.total), 0)
        downloaded = min(int(self.n), total) if total else int(self.n)
        pct = int((downloaded / total) * 100) if total else 0
        if not force and pct == self.last_pct:
            return
        self.last_pct = pct
        elapsed = max(now - self.started_at, 1e-9)
        rate = downloaded / elapsed
        rate_text = f"{_format_bytes(rate)}/s" if downloaded else "starting..."
        total_text = _format_bytes(total) if total else "unknown"
        current_text = _format_bytes(downloaded)
        filled = int((pct / 100) * 30) if total else 0
        bar = "#" * filled + "-" * (30 - filled)
        message = f"[{bar}] {pct:3d}% | {current_text} / {total_text} | {rate_text}"
        sys.stdout.write("\r" + message)
        sys.stdout.flush()

    def update(self, amount):
        self.n += amount
        self._render()

    def reset(self):
        self.n = 0
        self.started_at = time.time()
        self.last_pct = -1
        self._render(force=True)

    def close(self):
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()


def configure_moabb(mne_data_dir=MNE_DATA_DIR):
    import mne
    import moabb
    import pooch.downloaders as pooch_downloaders

    moabb.set_log_level("info")
    mne.set_log_level("info")
    pooch_downloaders.tqdm = NotebookDownloadProgress

    mne_data_path = Path(mne_data_dir).resolve()
    mne_data_path.mkdir(parents=True, exist_ok=True)
    mne.set_config("MNE_DATA", str(mne_data_path))
    moabb.set_download_dir(str(mne_data_path))


def get_cache_files(data_dir=PROCESSED_DATA_DIR):
    return sorted(
        Path(data_dir).glob("subject_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )


def download_subjects(data_dir=PROCESSED_DATA_DIR, mne_data_dir=MNE_DATA_DIR):
    configure_moabb(mne_data_dir=mne_data_dir)
    from moabb.datasets import Cho2017
    from moabb.paradigms import MotorImagery

    dataset = Cho2017()
    paradigm = MotorImagery(fmin=8, fmax=32)

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print("Checking and downloading missing subjects...")

    for subject in dataset.subject_list:
        file_path = Path(data_dir) / f"subject_{subject}.pt"
        if file_path.exists():
            continue

        print("\n=========================================")
        print(f"   Downloading and Processing Subject {subject}  ")
        print("=========================================")
        try:
            dataset.data_path(subject)
            X_sub, y_sub, _ = paradigm.get_data(dataset=dataset, subjects=[subject])
            torch.save((X_sub, y_sub), file_path)
            del X_sub, y_sub, _
            gc.collect()
        except Exception as exc:
            print(f"Skipping Subject {subject} due to download/processing error: {exc}")

    print("\nAll downloads complete or cached.")


def inspect_subject(subject=1, data_dir=PROCESSED_DATA_DIR):
    sample_file = Path(data_dir) / f"subject_{subject}.pt"
    print(f"--- Inspecting {sample_file} ---")
    try:
        X_sub, y_sub = torch.load(sample_file, weights_only=False)
    except FileNotFoundError:
        print(f"File not found: {sample_file}. Run the data preparation first.")
        return

    print("\n1. Data Structure:")
    print(f"X_sub (EEG Data) type: {type(X_sub)}")
    print(f"y_sub (Labels) type: {type(y_sub)}")

    print("\n2. Data Shapes:")
    print(
        f"X_sub shape: {X_sub.shape} -> "
        f"({X_sub.shape[0]} Epochs, {X_sub.shape[1]} Channels, {X_sub.shape[2]} Timepoints)"
    )
    print(f"y_sub length: {len(y_sub)} labels")

    print("\n3. Classification Values (Labels) & Epoch Counts:")
    for label, count in Counter(y_sub).items():
        print(f"  - Class '{label}': {count} epochs")


def truncate_subjects(data_dir=PROCESSED_DATA_DIR, max_epochs=200):
    print(f"Checking subjects and filtering to {max_epochs} epochs...")
    for file_path in get_cache_files(data_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        try:
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            num_epochs = len(y_sub)

            if num_epochs > max_epochs:
                print(f"Subject {subject_num}: Found {num_epochs} epochs. Truncating to {max_epochs}...")
                torch.save((X_sub[:max_epochs], y_sub[:max_epochs]), file_path)
            elif num_epochs < max_epochs:
                print(f"Subject {subject_num}: Has only {num_epochs} epochs.")
        except Exception as exc:
            print(f"Error loading Subject {subject_num}: {exc}")


def downsample_subjects(data_dir=PROCESSED_DATA_DIR, downsample_ratio=5, min_timepoints=400):
    print(f"Downsampling subjects by a ratio of {downsample_ratio}...")
    for file_path in get_cache_files(data_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        try:
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            original_timepoints = X_sub.shape[2]

            if original_timepoints < min_timepoints:
                print(
                    f"Subject {subject_num}: Skipping downsampling because it has "
                    f"only {original_timepoints} timepoints."
                )
                continue

            X_sub = X_sub[:, :, ::downsample_ratio]
            torch.save((X_sub, y_sub), file_path)
            print(f"Subject {subject_num}: Downsampled {original_timepoints} -> {X_sub.shape[2]} timepoints.")
        except Exception as exc:
            print(f"Error downsampling Subject {subject_num}: {exc}")

    print("Downsampling complete.")


def _load_gedai():
    try:
        from pygedai import gedai as run_gedai

        return run_gedai
    except (ImportError, AttributeError):
        try:
            from pygedai.GEDAI import gedai as run_gedai

            return run_gedai
        except (ImportError, AttributeError):
            return None


def apply_pygedai_preprocessing(
    data_dir=PROCESSED_DATA_DIR,
    output_dir=PYGEDAI_DATA_DIR,
    sfreq=CHO2017_SFREQ / 5,
):
    run_gedai = _load_gedai()
    if run_gedai is None:
        print("Skipping pygedai preprocessing: could not import pygedai.gedai.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Applying pygedai preprocessing to subjects...")
    for file_path in get_cache_files(data_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        try:
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            if not isinstance(X_sub, torch.Tensor):
                X_sub = torch.tensor(X_sub, dtype=torch.float32)

            cleaned_epochs = []
            n_channels = X_sub.shape[1]
            dummy_leadfield = torch.eye(n_channels, dtype=torch.float32)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for epoch_data in X_sub:
                    out = run_gedai(epoch_data, sfreq=sfreq, leadfield=dummy_leadfield)
                    cleaned_data = out["cleaned"]
                    if not isinstance(cleaned_data, torch.Tensor):
                        cleaned_data = torch.tensor(cleaned_data, dtype=torch.float32)
                    cleaned_epochs.append(cleaned_data)

            torch.save((torch.stack(cleaned_epochs), y_sub), output_path / file_path.name)
            print(f"Subject {subject_num}: Pygedai preprocessing saved to {output_path / file_path.name}.")
        except Exception as exc:
            print(f"Error applying pygedai to Subject {subject_num}: {exc}")


def prepare_data(
    data_dir=PROCESSED_DATA_DIR,
    mne_data_dir=MNE_DATA_DIR,
    max_epochs=200,
    downsample_ratio=5,
    min_timepoints=400,
    apply_pygedai=False,
    pygedai_output_dir=PYGEDAI_DATA_DIR,
):
    download_subjects(data_dir=data_dir, mne_data_dir=mne_data_dir)
    inspect_subject(subject=1, data_dir=data_dir)
    truncate_subjects(data_dir=data_dir, max_epochs=max_epochs)
    downsample_subjects(data_dir=data_dir, downsample_ratio=downsample_ratio, min_timepoints=min_timepoints)
    if apply_pygedai:
        apply_pygedai_preprocessing(data_dir=data_dir, output_dir=pygedai_output_dir)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Cho2017 MOABB EEG data.")
    parser.add_argument("--data-dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--mne-data-dir", default=MNE_DATA_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--downsample-ratio", type=int, default=5)
    parser.add_argument("--min-timepoints", type=int, default=400)
    parser.add_argument("--apply-pygedai", action="store_true")
    parser.add_argument("--pygedai-output-dir", default=PYGEDAI_DATA_DIR)
    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        mne_data_dir=args.mne_data_dir,
        max_epochs=args.max_epochs,
        downsample_ratio=args.downsample_ratio,
        min_timepoints=args.min_timepoints,
        apply_pygedai=args.apply_pygedai,
        pygedai_output_dir=args.pygedai_output_dir,
    )


if __name__ == "__main__":
    main()
