import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_MNE_DATA_DIR = Path("mne_data")
FIELDS = (
    "imagery_left",
    "imagery_right",
    "movement_left",
    "movement_right",
)


def subject_number_from_path(path):
    match = re.fullmatch(r"s(\d+)", path.stem.lower())
    if match:
        return int(match.group(1))

    match = re.fullmatch(r"subject_(\d+)", path.stem.lower())
    if match:
        return int(match.group(1))

    return None


def search_roots(mne_data_dir=DEFAULT_MNE_DATA_DIR):
    roots = []
    for root in (Path(mne_data_dir), Path.cwd()):
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def find_subject_mat(subject=1, mne_data_dir=DEFAULT_MNE_DATA_DIR):
    subject_names = [
        f"subject_{subject}.mat",
        f"s{subject:02d}.mat",
        f"s{subject}.mat",
    ]

    for root in search_roots(mne_data_dir):
        for name in subject_names:
            direct_path = root / name
            if direct_path.exists():
                return direct_path

        for name in subject_names:
            matches = sorted(root.rglob(name))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Could not find subject {subject} MAT file. Looked for: {', '.join(subject_names)}"
    )


def find_all_subject_mats(mne_data_dir=DEFAULT_MNE_DATA_DIR):
    subjects = {}

    for root in search_roots(mne_data_dir):
        for mat_path in sorted(root.rglob("*.mat")):
            subject = subject_number_from_path(mat_path)
            if subject is not None and subject not in subjects:
                subjects[subject] = mat_path

    if not subjects:
        raise FileNotFoundError(f"Could not find any subject MAT files under {mne_data_dir}.")

    return [(subject, subjects[subject]) for subject in sorted(subjects)]


def load_eeg(mat_path):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "eeg" not in mat:
        raise KeyError(f"{mat_path} does not contain an 'eeg' variable.")
    return mat["eeg"]


def epoch_samples(eeg):
    if not hasattr(eeg, "frame") or not hasattr(eeg, "srate"):
        return None

    frame = np.asarray(eeg.frame, dtype=float)
    if frame.size != 2:
        return None

    duration_seconds = (frame.max() - frame.min()) / 1000.0
    return int(round(duration_seconds * float(eeg.srate)))


def count_epochs(field_value, samples_per_epoch=None):
    data = np.asarray(field_value)
    if samples_per_epoch and data.ndim >= 2 and data.shape[-1] % samples_per_epoch == 0:
        return data.shape[-1] // samples_per_epoch
    if data.ndim == 0:
        return 1
    return data.shape[0]


def subject_epoch_breakdown(mat_path):
    eeg = load_eeg(mat_path)
    samples_per_epoch = epoch_samples(eeg)
    counts = {}

    for field in FIELDS:
        if not hasattr(eeg, field):
            counts[field] = 0
            continue
        counts[field] = count_epochs(getattr(eeg, field), samples_per_epoch)

    return counts, samples_per_epoch


def print_breakdown_table(rows):
    headers = ["subject", "samples_per_epoch", *FIELDS]
    widths = {header: len(header) for header in headers}

    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    divider = "  ".join("-" * widths[header] for header in headers)
    print(header_line)
    print(divider)

    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main():
    parser = argparse.ArgumentParser(
        description="Count motor imagery/movement epochs in subject MAT files."
    )
    parser.add_argument("--subject", default="all", help="Use 'all' or a subject number like 1.")
    parser.add_argument("--mne-data-dir", default=str(DEFAULT_MNE_DATA_DIR))
    parser.add_argument("--mat-path", default=None)
    args = parser.parse_args()

    if args.mat_path:
        mat_paths = [(subject_number_from_path(Path(args.mat_path)) or args.subject, Path(args.mat_path))]
    elif args.subject.lower() == "all":
        mat_paths = find_all_subject_mats(mne_data_dir=Path(args.mne_data_dir))
    else:
        subject = int(args.subject)
        mat_paths = [(subject, find_subject_mat(subject=subject, mne_data_dir=Path(args.mne_data_dir)))]

    rows = []
    for subject, mat_path in mat_paths:
        counts, samples_per_epoch = subject_epoch_breakdown(mat_path)
        row = {
            "subject": subject,
            "samples_per_epoch": samples_per_epoch or "unknown",
        }
        row.update(counts)
        rows.append(row)

    print_breakdown_table(rows)


if __name__ == "__main__":
    main()
