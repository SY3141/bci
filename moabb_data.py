import argparse
import gc
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import urllib3
from scipy.signal import resample_poly


PROCESSED_DATA_DIR = "processed_data"
PYGEDAI_DATA_DIR = "pygedai_processed"
DOWNSAMPLED_DATA_DIR = "downsampled"
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


def free_cached_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_subject_cache_readable(file_path):
    X_sub, y_sub = None, None
    try:
        X_sub, y_sub = torch.load(file_path, weights_only=False)
        _ = X_sub.shape
        _ = len(y_sub)
        return True
    except Exception as exc:
        print(f"Cached file {file_path} is unreadable and will be redownloaded: {exc}")
        return False
    finally:
        del X_sub, y_sub
        free_cached_memory()


def _event_id_values(event_id):
    values = []
    for value in event_id.values():
        values.extend(np.atleast_1d(value).tolist())
    return values


def _event_label_lookup(event_id):
    lookup = {}
    for label, value in event_id.items():
        for code in np.atleast_1d(value).tolist():
            lookup[int(code)] = label
    return lookup


def _extract_original_subject_epochs(dataset, subject):
    import mne

    subject_data = dataset.get_data(subjects=[subject])[subject]
    event_id = dataset.event_id
    event_values = _event_id_values(event_id)
    label_lookup = _event_label_lookup(event_id)

    X_runs, y_runs = [], []
    for session_data in subject_data.values():
        for raw in session_data.values():
            stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
            if stim_channels:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

            events = events[np.isin(events[:, 2], event_values)]
            if len(events) == 0:
                continue

            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=dataset.interval[0],
                tmax=dataset.interval[1],
                baseline=None,
                proj=False,
                preload=True,
                picks="eeg",
                event_repeated="drop",
                on_missing="ignore",
                verbose=False,
            )

            X_runs.append(epochs.get_data())
            y_runs.extend(label_lookup[int(code)] for code in epochs.events[:, 2])

    if not X_runs:
        raise ValueError(f"No usable EEG epochs found for subject {subject}.")

    return np.concatenate(X_runs, axis=0), np.asarray(y_runs)


def download_subjects(data_dir=PROCESSED_DATA_DIR, mne_data_dir=MNE_DATA_DIR):
    configure_moabb(mne_data_dir=mne_data_dir)
    from moabb.datasets import Cho2017

    dataset = Cho2017()

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print("Checking and downloading missing subjects...")

    for subject in dataset.subject_list:
        file_path = Path(data_dir) / f"subject_{subject}.pt"
        if file_path.exists():
            if is_subject_cache_readable(file_path):
                continue
            try:
                file_path.unlink()
            except OSError as exc:
                print(f"Could not remove corrupted cache file {file_path}: {exc}")
                continue

        print("\n=========================================")
        print(f"   Downloading and Processing Subject {subject}  ")
        print("=========================================")
        X_sub, y_sub = None, None
        try:
            dataset.data_path(subject)
            X_sub, y_sub = _extract_original_subject_epochs(dataset, subject)
            torch.save((X_sub, y_sub), file_path)
        except Exception as exc:
            print(f"Skipping Subject {subject} due to download/processing error: {exc}")
        finally:
            del X_sub, y_sub
            free_cached_memory()

    print("\nAll downloads complete or cached.")


def inspect_subject(subject=1, data_dir=PROCESSED_DATA_DIR):
    sample_file = Path(data_dir) / f"subject_{subject}.pt"
    print(f"--- Inspecting {sample_file} ---")
    X_sub, y_sub = None, None
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
    del X_sub, y_sub
    free_cached_memory()


def truncate_subjects(data_dir=PROCESSED_DATA_DIR, max_epochs=200):
    print(f"Checking subjects and filtering to {max_epochs} epochs...")
    for file_path in get_cache_files(data_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        X_sub, y_sub = None, None
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
        finally:
            del X_sub, y_sub
            free_cached_memory()


def downsample_subjects(
    input_dir=PYGEDAI_DATA_DIR,
    output_dir=DOWNSAMPLED_DATA_DIR,
    downsample_ratio=4,
    min_timepoints=400,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downsampling subjects by a ratio of {downsample_ratio} into {output_path}...")
    for file_path in get_cache_files(input_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        X_sub, y_sub = None, None
        try:
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            original_timepoints = X_sub.shape[2]
            subject_output_path = output_path / file_path.name

            if original_timepoints < min_timepoints:
                torch.save((X_sub, y_sub), subject_output_path)
                print(
                    f"Subject {subject_num}: Skipping downsampling because it has "
                    f"only {original_timepoints} timepoints. Copied to {subject_output_path}."
                )
                continue

            X_sub = resample_poly(X_sub, up=1, down=downsample_ratio, axis=-1)
            X_sub = torch.as_tensor(X_sub, dtype=torch.float32)
            torch.save((X_sub, y_sub), subject_output_path)
            print(
                f"Subject {subject_num}: Downsampled {original_timepoints} -> "
                f"{X_sub.shape[2]} timepoints, saved to {subject_output_path}."
            )
        except Exception as exc:
            print(f"Error downsampling Subject {subject_num}: {exc}")
        finally:
            del X_sub, y_sub
            free_cached_memory()

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


def _load_interpolate_ref_cov():
    try:
        from pygedai import interpolate_ref_cov

        return interpolate_ref_cov
    except (ImportError, AttributeError):
        try:
            from pygedai.ref_cov import interpolate_ref_cov

            return interpolate_ref_cov
        except (ImportError, AttributeError):
            return None


def build_cho2017_reference_covariance(mne_data_dir=MNE_DATA_DIR, dtype=torch.float32):
    configure_moabb(mne_data_dir=mne_data_dir)

    import mne
    import pandas as pd
    from moabb.datasets import Cho2017

    interpolate_ref_cov = _load_interpolate_ref_cov()
    if interpolate_ref_cov is None:
        raise ImportError("Could not import pygedai.interpolate_ref_cov.")

    dataset = Cho2017()
    subject_data = dataset.get_data(subjects=[1])[1]
    session_key = next(iter(subject_data))
    run_key = next(iter(subject_data[session_key]))
    print(f"Using Cho2017 subject 1, session '{session_key}', run '{run_key}' for channel names.")
    raw = subject_data[session_key][run_key]
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[i] for i in eeg_picks]

    montage = mne.channels.make_standard_montage("standard_1005")
    positions = montage.get_positions()["ch_pos"]
    position_lookup = {name.lower(): xyz for name, xyz in positions.items()}

    rows = []
    missing_channels = []
    for ch_name in ch_names:
        if ch_name in positions:
            xyz = positions[ch_name]
            rows.append({"channel_name": ch_name, "X": xyz[0], "Y": xyz[1], "Z": xyz[2]})
        elif ch_name.lower() in position_lookup:
            xyz = position_lookup[ch_name.lower()]
            rows.append({"channel_name": ch_name, "X": xyz[0], "Y": xyz[1], "Z": xyz[2]})
        else:
            missing_channels.append(ch_name)

    if missing_channels:
        print(f"Warning: Coordinates not found in standard_1005 for: {missing_channels}")

    if not rows:
        raise ValueError("No Cho2017 EEG channels could be matched to standard_1005 coordinates.")

    electrode_positions = pd.DataFrame(rows, columns=["channel_name", "X", "Y", "Z"])
    print("Interpolating PyGEDAI reference covariance from Cho2017 standard_1005 channel coordinates...")
    reference_covariance = interpolate_ref_cov(electrode_positions, dtype=dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_covariance = reference_covariance.to(device)
    print(f"PyGEDAI reference covariance shape: {tuple(reference_covariance.shape)} on {device}")
    return reference_covariance, device


def apply_pygedai_preprocessing(
    data_dir=PROCESSED_DATA_DIR,
    output_dir=PYGEDAI_DATA_DIR,
    sfreq=CHO2017_SFREQ,
    mne_data_dir=MNE_DATA_DIR,
):
    run_gedai = _load_gedai()
    if run_gedai is None:
        print("Skipping pygedai preprocessing: could not import pygedai.gedai.")
        return

    try:
        reference_covariance, device = build_cho2017_reference_covariance(
            mne_data_dir=mne_data_dir,
            dtype=torch.float32,
        )
    except Exception as exc:
        print(f"Skipping pygedai preprocessing: could not build Cho2017 reference covariance. e={exc}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Applying pygedai preprocessing to subjects...")
    for file_path in get_cache_files(data_dir):
        subject_num = int(file_path.stem.split("_")[-1])
        X_sub, y_sub, cleaned_subject, cleaned_data, epoch_data, out = None, None, None, None, None, None
        try:
            X_sub, y_sub = torch.load(file_path, weights_only=False)
            if not isinstance(X_sub, torch.Tensor):
                X_sub = torch.as_tensor(X_sub, dtype=torch.float32)
            else:
                X_sub = X_sub.to(dtype=torch.float32, device="cpu")

            cleaned_subject = torch.empty_like(X_sub, dtype=torch.float32, device="cpu")
            n_channels = X_sub.shape[1]
            if reference_covariance.shape != (n_channels, n_channels):
                raise ValueError(
                    f"Reference covariance shape {tuple(reference_covariance.shape)} does not match "
                    f"subject tensor channel count {n_channels}."
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for epoch_idx, epoch_data in enumerate(X_sub):
                    with torch.no_grad():
                        out = run_gedai(
                            epoch_data,
                            sfreq=sfreq,
                            leadfield=reference_covariance,
                            device=device,
                            dtype=torch.float32,
                        )
                    cleaned_data = out["cleaned"]
                    if not isinstance(cleaned_data, torch.Tensor):
                        cleaned_data = torch.as_tensor(cleaned_data, dtype=torch.float32)
                    cleaned_data = cleaned_data.detach().to(device="cpu", dtype=torch.float32)
                    cleaned_subject[epoch_idx].copy_(cleaned_data)

                    del cleaned_data, out
                    cleaned_data, out = None, None
                    if (epoch_idx + 1) % 10 == 0:
                        free_cached_memory()

            torch.save((cleaned_subject, y_sub), output_path / file_path.name)
            print(f"Subject {subject_num}: Pygedai preprocessing saved to {output_path / file_path.name}.")
        except Exception as exc:
            print(f"Error applying pygedai to Subject {subject_num}: {exc}")
        finally:
            del X_sub, y_sub, cleaned_subject, cleaned_data, epoch_data, out
            free_cached_memory()

    del reference_covariance
    free_cached_memory()


def prepare_data(
    data_dir=PROCESSED_DATA_DIR,
    mne_data_dir=MNE_DATA_DIR,
    downsampled_dir=DOWNSAMPLED_DATA_DIR,
    max_epochs=200,
    downsample_ratio=4,
    min_timepoints=400,
):
    download_subjects(data_dir=data_dir, mne_data_dir=mne_data_dir)
    inspect_subject(subject=1, data_dir=data_dir)
    truncate_subjects(data_dir=data_dir, max_epochs=max_epochs)
    apply_pygedai_preprocessing(data_dir=data_dir, output_dir=PYGEDAI_DATA_DIR, mne_data_dir=mne_data_dir)
    downsample_subjects(
        input_dir=PYGEDAI_DATA_DIR,
        output_dir=downsampled_dir,
        downsample_ratio=downsample_ratio,
        min_timepoints=min_timepoints,
    )


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Cho2017 MOABB EEG data.")
    parser.add_argument("--data-dir", default=PROCESSED_DATA_DIR)
    parser.add_argument("--mne-data-dir", default=MNE_DATA_DIR)
    parser.add_argument("--downsampled-dir", default=DOWNSAMPLED_DATA_DIR)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--downsample-ratio", type=int, default=4)
    parser.add_argument("--min-timepoints", type=int, default=400)
    args = parser.parse_args()

    prepare_data(
        data_dir=args.data_dir,
        mne_data_dir=args.mne_data_dir,
        downsampled_dir=args.downsampled_dir,
        max_epochs=args.max_epochs,
        downsample_ratio=args.downsample_ratio,
        min_timepoints=args.min_timepoints,
    )


if __name__ == "__main__":
    main()
