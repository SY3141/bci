from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BENCHMARK_CSV = Path("benchmark.csv")
LORA_BENCHMARK_CSV = Path("lora_benchmark.csv")
OUTPUT_PATH = Path("boxplot.png")
BENCHMARK_ACCURACY_COLUMN = "validation_accuracy"
LORA_ACCURACY_COLUMN = "test_accuracy"
SUBJECT_COLUMN = "subject"


def load_accuracy_by_subject(csv_path, accuracy_column):
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}.")

    df = pd.read_csv(csv_path)
    if accuracy_column not in df.columns:
        raise ValueError(f"{csv_path} does not contain a '{accuracy_column}' column.")
    if SUBJECT_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} does not contain a '{SUBJECT_COLUMN}' column.")

    values = df[[SUBJECT_COLUMN, accuracy_column]].copy()
    values[accuracy_column] = pd.to_numeric(values[accuracy_column], errors="coerce")
    values = values.dropna(subset=[accuracy_column])
    if values.empty:
        raise ValueError(f"{csv_path} does not contain any numeric accuracy values.")

    return values


def main():
    benchmark = load_accuracy_by_subject(BENCHMARK_CSV, BENCHMARK_ACCURACY_COLUMN).rename(
        columns={BENCHMARK_ACCURACY_COLUMN: "benchmark_accuracy"}
    )
    lora = load_accuracy_by_subject(LORA_BENCHMARK_CSV, LORA_ACCURACY_COLUMN).rename(
        columns={LORA_ACCURACY_COLUMN: "lora_accuracy"}
    )
    paired = benchmark.merge(
        lora,
        on=SUBJECT_COLUMN,
    ).sort_values(SUBJECT_COLUMN)

    if paired.empty:
        raise ValueError("No matching subjects found between benchmark CSVs.")

    benchmark_accuracy = paired["benchmark_accuracy"]
    lora_accuracy = paired["lora_accuracy"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(
        [benchmark_accuracy, lora_accuracy],
        labels=["Benchmark validation", "LoRA test"],
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#9ecae1", "edgecolor": "#2f4858"},
        medianprops={"color": "#d1495b", "linewidth": 2},
        whiskerprops={"color": "#2f4858"},
        capprops={"color": "#2f4858"},
    )

    for _, row in paired.iterrows():
        y_values = [
            row["benchmark_accuracy"],
            row["lora_accuracy"],
        ]
        ax.plot([1, 2], y_values, color="#6c757d", alpha=0.45, linewidth=1)

    ax.scatter(
        [1] * len(paired),
        benchmark_accuracy,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        s=42,
        zorder=3,
        label="Benchmark validation subject",
    )
    ax.scatter(
        [2] * len(paired),
        lora_accuracy,
        color="#e76f51",
        edgecolor="white",
        linewidth=0.8,
        s=42,
        zorder=3,
        label="LoRA test subject",
    )
    ax.set_title("Benchmark Validation vs LoRA Test Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300)
    print(f"Saved box plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
