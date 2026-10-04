"""Exploratory Data Analysis: saves plots to screenshots/eda/.

Usage:
    python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (works in CI/Docker)
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    get_clean_data,
)

OUT_DIR = Path("screenshots/eda")
LABELS = {0: "No disease", 1: "Disease"}
sns.set_theme(style="whitegrid", palette="deep")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_DIR / name}")


def plot_class_balance(df) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET].map(LABELS).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, f"{v} ({v / len(df):.1%})", ha="center")
    ax.set(title="Class balance", xlabel="", ylabel="Patients")
    save(fig, "class_balance.png")


def plot_histograms(df) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=df[TARGET].map(LABELS),
                     kde=True, bins=20, ax=ax)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    save(fig, "numeric_histograms.png")


def plot_boxplots(df) -> None:
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(x=df[TARGET].map(LABELS), y=df[col], ax=ax)
        ax.set(title=col, xlabel="")
    save(fig, "numeric_boxplots_by_target.png")


def plot_categorical(df) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=df[TARGET].map(LABELS), ax=ax)
        ax.set_title(col)
        ax.legend(title="", fontsize=8)
    save(fig, "categorical_counts_by_target.png")


def plot_correlation(df) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "correlation_heatmap.png")

    # Features ranked by correlation with the target
    fig, ax = plt.subplots(figsize=(6, 6))
    tc = corr[TARGET].drop(TARGET).sort_values()
    sns.barplot(x=tc.values, y=tc.index, ax=ax)
    ax.set(title="Correlation with target", xlabel="Pearson r")
    save(fig, "target_correlation.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = get_clean_data()

    print("Shape:", df.shape)
    print("\nSummary statistics:\n", df.describe().T.round(2))
    print("\nClass distribution:\n", df[TARGET].value_counts(normalize=True).round(3))

    plot_class_balance(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_categorical(df)
    plot_correlation(df)


if __name__ == "__main__":
    main()
