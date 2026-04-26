import numpy as np
import os
import torch
import warnings
import urllib3
import gc
import time
import sys
import moabb
from moabb.datasets import Cho2017
from moabb.paradigms import MotorImagery
import mne
import pooch.downloaders as pooch_downloaders

# Force verbose logging for MOABB and MNE.
moabb.set_log_level('info')
mne.set_log_level('info')

# Suppress InsecureRequestWarning for HTTP downloads
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

def _format_bytes(num_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != 'B' else f"{int(value)} {unit}"
        value /= 1024

class NotebookDownloadProgress:
    def __init__(self, total=0, unit='B', unit_scale=True, **kwargs):
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
        total_text = _format_bytes(total) if total else 'unknown'
        current_text = _format_bytes(downloaded)
        filled = int((pct / 100) * 30) if total else 0
        bar = '#' * filled + '-' * (30 - filled)
        message = f"[{bar}] {pct:3d}% | {current_text} / {total_text} | {rate_text}"
        sys.stdout.write('\r' + message)
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
        sys.stdout.write('\n')
        sys.stdout.flush()

# Pooch defaults to terminal-style tqdm output, which renders as raw ANSI in Jupyter.
# Replacing its tqdm factory gives us a notebook-safe progress UI for MOABB downloads.
pooch_downloaders.tqdm = NotebookDownloadProgress

# Set MNE data path to an absolute local directory so it caches raw downloads predictably on Windows.
mne_data_dir = os.path.abspath(os.path.join(os.getcwd(), 'mne_data'))
os.makedirs(mne_data_dir, exist_ok=True)
mne.set_config('MNE_DATA', mne_data_dir)
moabb.set_download_dir(mne_data_dir)  # FORCE moabb to use the local directory

# 1. Initialize the dataset & paradigm
dataset = Cho2017()
paradigm = MotorImagery(fmin=8, fmax=32)

# 2. Iterate through each subject to download and save locally (if not cached)
os.makedirs('processed_data', exist_ok=True)

print("Checking and downloading missing subjects...")

# Download each missing subject individually so notebook progress stays visible and cached files are reused.
for sub in dataset.subject_list:
    file_path = f"processed_data/subject_{sub}.pt"
    
    if not os.path.exists(file_path):
        print(f"\n=========================================")
        print(f"   Downloading and Processing Subject {sub}  ")
        print(f"=========================================")
        try:
            # Trigger the download explicitly so the notebook-safe Pooch progress bar is shown first.
            dataset.data_path(sub)
            
            # Fetch data for just this subject (instantly uses the downloaded files)
            X_sub, y_sub, _ = paradigm.get_data(dataset=dataset, subjects=[sub])
            # Save the extracted epoch numpy arrays to a local PyTorch file
            torch.save((X_sub, y_sub), file_path)
            
            # Explicitly force garbage collection to prevent memory-leak stalling 
            del X_sub, y_sub, _
            gc.collect()
            
        except Exception as e:
            print(f"Skipping Subject {sub} due to download/processing error: {e}")
            continue

print("\nAll downloads complete or cached.")