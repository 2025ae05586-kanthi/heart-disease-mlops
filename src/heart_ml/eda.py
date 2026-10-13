
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.heart_ml.data import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def run_eda():
    """Generate exploratory data analysis plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset()

    sns.set_theme(style="whitegrid")

    # 1. Target class distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Target Class Distribution")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "class_balance.png",
        dpi=150,
    )
    plt.close()

    # 2. Feature distributions
    features = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 8),
    )

    for ax, feature in zip(axes.flat, features):
        sns.histplot(
            data=df,
            x=feature,
            hue="target",
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feature}")

    # Hide the unused sixth subplot.
    axes.flat[-1].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "feature_distributions.png",
        dpi=150,
    )
    plt.close(fig)

    # 3. Correlation heatmap
    plt.figure(figsize=(12, 9))
    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=False,
    )

    plt.title("Feature and Target Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close()

    # 4. Age distribution by target
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="target",
        y="age",
    )

    plt.title("Age Distribution by Target Class")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "age_by_target.png",
        dpi=150,
    )
    plt.close()

    # 5. Missing-value summary
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        plt.figure(figsize=(7, 4))
        missing.sort_values().plot(kind="bar")
        plt.title("Missing Values by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Missing value count")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "missing_values.png",
            dpi=150,
        )
        plt.close()

    print("\nEDA completed successfully.")
    print(f"Figures saved in: {FIGURES_DIR}")

    for figure in sorted(FIGURES_DIR.glob("*.png")):
        print(f" - {figure.name}")


if __name__ == "__main__":
    run_eda()