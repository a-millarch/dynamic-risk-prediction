import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from collections import OrderedDict

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.calibration import calibration_curve

from fastai.tabular.all import L
from math import sqrt


def plot_box_kde(df, dep, y):

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    pal = {1: sns.color_palette()[1], 0: sns.color_palette()[0]}
    sns.boxenplot(data=df, x=dep, y=y, ax=axs[0], palette=pal, hue=dep).legend(
        [], [], frameon=False
    )
    sns.kdeplot(data=df[df[dep] == 0], x=y, ax=axs[1], label="0", clip=(0, 1))
    sns.kdeplot(data=df[df[dep] == 1], x=y, ax=axs[1], label="1", clip=(0, 1))
    fig.legend()

    mask_0 = df[dep] == 0
    mask_1 = df[dep] == 1

    print(
        "is null in total\t\t", "%.2f" % float(df[y].isna().sum() / len(df) * 100), "%"
    )
    print(
        "is null in",
        dep,
        "== 0\t",
        "%.2f" % float(df[mask_0][y].isna().sum() / len(df[mask_0]) * 100),
        "%",
    )
    print(
        "is null in",
        dep,
        "== 1\t",
        "%.2f" % float(df[mask_1][y].isna().sum() / len(df[mask_1]) * 100),
        "%",
    )
    return fig


def plot_hist_kde(df, dep, y):

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    pal = {1: sns.color_palette()[1], 0: sns.color_palette()[0]}

    # Replace boxenplot with histplot
    sns.histplot(
        data=df, x=y, hue=dep, ax=axs[0], palette=pal, kde=True, stat="density"
    )
    axs[0].legend(title=dep)

    sns.kdeplot(data=df[df[dep] == 0], x=y, ax=axs[1], label="0", clip=(0, 1))
    sns.kdeplot(data=df[df[dep] == 1], x=y, ax=axs[1], label="1", clip=(0, 1))
    fig.legend()

    mask_0 = df[dep] == 0
    mask_1 = df[dep] == 1

    print(
        "is null in total\t\t", "%.2f" % float(df[y].isna().sum() / len(df) * 100), "%"
    )
    print(
        "is null in",
        dep,
        "== 0\t",
        "%.2f" % float(df[mask_0][y].isna().sum() / len(df[mask_0]) * 100),
        "%",
    )
    print(
        "is null in",
        dep,
        "== 1\t",
        "%.2f" % float(df[mask_1][y].isna().sum() / len(df[mask_1]) * 100),
        "%",
    )
    return fig


def plot_evaluation(y_preds, ys, target):

    fpr, tpr, thresholds = roc_curve(ys, y_preds)
    roc_auc = roc_auc_score(ys, y_preds)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{target}"
    ).plot(ax=ax[0], c="r")

    display2 = PrecisionRecallDisplay.from_predictions(
        ys, y_preds, ax=ax[1], figure=fig, name=f"{target}"
    )  # ,
    # legend=f'{target} within 12H post trauma reception')
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, c="grey")
    ax[0].set_title("Holdout:\nReceiver operator characteristics (ROC)")
    ax[0].grid()
    ax[1].plot([0, 1], [ys.sum() / len(ys), ys.sum() / len(ys)], "k--", lw=1, c="grey")
    ax[1].set_title("Holdout:\nPrecision Recall (PR)")
    ax[1].grid()
    plt.savefig("models/figs/holdout_evaluation.png", dpi=1200)
    plt.plot()
    return fig


def plot_loss(learn, fold=None):
    # Create a Figure and Axes object
    fig, ax = plt.subplots(figsize=(10, 5))

    skip_start = 0
    # Plot training losses
    train_iterations = list(range(skip_start, len(learn.recorder.losses)))
    train_losses = learn.recorder.losses[skip_start:]
    ax.plot(train_iterations, train_losses, label="Train", color="blue")

    # Calculate index for validation losses
    idx = (np.array(learn.recorder.iters) < skip_start).sum()
    valid_col = learn.recorder.metric_names.index("valid_loss") - 1

    # Plot validation losses
    valid_iterations = learn.recorder.iters[idx:]
    valid_losses = L(learn.recorder.values[idx:]).itemgot(valid_col)
    ax.plot(valid_iterations, valid_losses, label="Validation", color="orange")  # type: ignore

    # Set legend and labels
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    if fold is None:
        ax.set_title(f"Loss Plot")
        plt.savefig(f"models/figs/loss_plot.png")
    else:
        ax.set_title(f"Loss Plot for Fold {fold}")
        # Save the figure
        plt.savefig(
            f"models/figs/{fold}_fold_loss_plot.png"
        )  # Ensure to specify file extension

    # Show the plot
    plt.show()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "Iteration": train_iterations + valid_iterations,
            "Loss": train_losses + list(valid_losses),
            "Type": ["Train"] * len(train_losses) + ["Validation"] * len(valid_losses),
        }
    )

    return fig, df  # Return both the Figure object and the DataFrame


def plot_fold_evaluation(metrics, target):
    fprs = metrics["fpr"]["scores"]
    tprs = metrics["tpr"]["scores"]
    precisions = metrics["precision"]["scores"]
    recalls = metrics["recall"]["scores"]
    roc_aucs = metrics["roc_auc"]["scores"]
    aps = metrics["avg_precision"]["scores"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot ROC curves
    for i in range(len(fprs)):
        RocCurveDisplay(
            fpr=fprs[i], tpr=tprs[i], roc_auc=roc_aucs[i], estimator_name=f"Fold {i+1}"
        ).plot(ax=ax[0])
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, c="grey")
    ax[0].set_title(f"Receiver Operating Characteristic (ROC) - {target}")
    ax[0].grid(True)

    # Plot PR curves
    for i in range(len(precisions)):
        PrecisionRecallDisplay(
            precision=precisions[i],
            recall=recalls[i],
            average_precision=aps[i],
            estimator_name=f"Fold {i+1}",
        ).plot(ax=ax[1])
    ax[1].plot([0, 1], [0.5, 0.5], "k--", lw=1, c="grey")
    ax[1].set_title(f"Precision-Recall (PR) - {target}")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"models/figs/{target}_evaluation_plots")
    plt.show()
    return fig


def evaluate_detection_rate(y_preds, y_true, threshold=0.5):
    """
    Evaluate the detection rate by calculating sensitivity and specificity.

    Parameters:
    - y_preds: Predicted probabilities for the positive class.
    - y_true: True binary labels.
    - threshold: Probability threshold for classifying as positive (default: 0.5).
    """
    # Calculate the percentage of positive patients
    positive_percentage = np.mean(y_true) * 100
    print(f"Percentage of positive patients: {positive_percentage:.2f}%")

    # Generate predictions based on the threshold
    y_pred_risk = (y_preds >= threshold).astype(int)

    # Calculate confusion matrix using the entire dataset with normalization
    cm = confusion_matrix(y_true, y_pred_risk, normalize="true")

    # Extract true positives, true negatives, false positives, and false negatives
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred_risk).ravel()

    # Calculate sensitivity and specificity from the confusion matrix
    sensitivity = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0

    print(
        f"Sensitivity (% of truly positive patients classified as positive): {sensitivity:.2f}%"
    )
    print(
        f"Specificity (% of truly negative patients classified as negative): {specificity:.2f}%"
    )

    # Create subplots for confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute confusion matrix
    cm_absolute = confusion_matrix(y_true, y_pred_risk)
    ConfusionMatrixDisplay(
        confusion_matrix=cm_absolute, display_labels=["Negative", "Positive"]
    ).plot(ax=ax[0], values_format="d")
    ax[0].set_title("Confusion Matrix (Absolute)")

    # Normalized confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    ).plot(ax=ax[1], values_format=".2f")
    ax[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(f"models/figs/cm")
    plt.show()
    return fig


def create_calibration_plot(y_true, y_pred, n_bins=4):
    # Create a Figure and Axes object
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)

    # Plot model calibration curve
    ax.plot(prob_pred, prob_true, marker="o", label="Model")

    # Set labels and title
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Plot")
    ax.legend(loc="lower right")
    ax.grid(True)

    # Calculate and display Brier score
    brier_score = np.mean((y_pred - y_true) ** 2)
    ax.text(0.1, 0.9, f"Brier Score: {brier_score:.4f}", transform=ax.transAxes)

    # Save the figure
    plt.savefig("models/figs/calibration.png")  # Ensure to specify file extension

    # Show the plot
    plt.show()

    return fig  # Return the Figure object
