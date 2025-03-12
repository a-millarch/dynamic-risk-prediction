import os
import numpy as np
import pandas as pd
import json

from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch

import mlflow


def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        json.dump(metrics, f, cls=NumpyEncoder)


# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_threshold(y_true, y_score, method="F-score", beta=None):
    """Find data-driven cut-off for classification
        Cut-off is determied using F-score

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).

    """
    if method == "Youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx]
    #     print('Best Threshold: {}'.format(threshold))

    elif method == "F-score":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        df_recall_precision = pd.DataFrame(
            {
                "Precision": precision[:-1],
                "Recall": recall[:-1],
                "Threshold": thresholds,
            }
        )
        fscore = (2 * precision * recall) / (precision + recall)
        # Find the optimal threshold
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits=4)
        fscoreOpt = round(fscore[index], ndigits=4)
        recallOpt = round(recall[index], ndigits=4)
        precisionOpt = round(precision[index], ndigits=4)
        #   print('Best Threshold: {}'.format(thresholdOpt))

        threshold = thresholdOpt

    elif method == "Fbeta-score":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        df_recall_precision = pd.DataFrame(
            {
                "Precision": precision[:-1],
                "Recall": recall[:-1],
                "Threshold": thresholds,
            }
        )
        if beta is None:
            beta = 2
        fscore = ((1 + pow(beta, 2)) * precision * recall) / (
            pow(beta, 2) * precision + recall
        )

        # Find the optimal threshold
        index = np.argmax(fscore)
        thresholdOpt = round(thresholds[index], ndigits=4)
        fscoreOpt = round(fscore[index], ndigits=4)
        recallOpt = round(recall[index], ndigits=4)
        precisionOpt = round(precision[index], ndigits=4)
        # print('Best Threshold: {}'.format(thresholdOpt))
        # print('Recall: {}, Precision: {}'.format(recallOpt, precisionOpt))

        threshold = thresholdOpt

    return threshold


def get_class_weights(df, target):
    """
    Returns class weight from dataframe with binary dependent variable(target)

                wj=n_samples / (n_classes * n_samplesj)
    where:
        wj is the weight for each class(j signifies the class)
        n_samplesis the total number of samples or rows in the dataset
        n_classesis the total number of unique classes in the target
        n_samplesjis the total number of rows of the respective class
    """
    class_count_df = df.groupby(target).count()
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    w_0 = (n_0 + n_1) / (2.0 * n_0)
    w_1 = (n_0 + n_1) / (2.0 * n_1)
    class_weights = torch.FloatTensor([w_0, w_1])

    print("[0,1] weighted as ", class_weights)
    return class_weights


def log_figures(figs_dir):
    """Log all figures in the specified directory to MLflow."""
    for filename in os.listdir(figs_dir):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            # Replace 'path/to/image.png' with the actual path to your image
            img = mpimg.imread(f"{figs_dir}/{filename}")
            plt.imshow(img)
            plt.axis("off")
            # fig_path = os.path.join(plt, filename)
            mlflow.log_figure(plt, filename)


def initialize_metrics():
    return {
        "roc_auc": {"best_score": 0, "best_model": None, "best_fold": 0, "scores": []},
        "avg_precision": {
            "best_score": 0,
            "best_model": None,
            "best_fold": 0,
            "scores": [],
        },
        "meta": {"best_epoch": []},
        "precision": {"scores": []},
        "recall": {"scores": []},
        "fpr": {"scores": []},
        "tpr": {"scores": []},
    }


def initialize_fmetrics():
    fmetrics = {}
    for beta in range(1, 21):
        fmetrics[f"f{beta}"] = {
            "best_score": 0,
            "best_model": None,
            "best_fold": 0,
            "scores": [],
            "optimal_thresholds": [],
        }
    return fmetrics


def calculate_fmetrics_stats(fmetrics):
    fmetrics_stats = {}
    for beta in range(1, 21):
        thresholds = fmetrics[f"f{beta}"]["optimal_thresholds"]
        fmetrics_stats[f"f{beta}"] = {
            "threshold_mean": np.mean(thresholds),
            "threshold_std": np.std(thresholds),
            "best_score": fmetrics[f"f{beta}"]["best_score"],
            "best_fold": fmetrics[f"f{beta}"]["best_fold"],
        }
    return fmetrics_stats


def evaluate_and_log_fmetrics(preds, target, fmetrics, fold=None, learn=None):
    precision, recall, thresholds = precision_recall_curve(target, preds)

    for beta in range(1, 21):
        fbetascore = [
            ((1 + beta**2) * p * r) / ((beta**2 * p) + r) if p + r > 0 else 0
            for p, r in zip(precision, recall)
        ]
        max_fbetascore = max(fbetascore)
        optimal_threshold = thresholds[fbetascore.index(max_fbetascore)]

        fmetrics[f"f{beta}"]["scores"].append(max_fbetascore)
        fmetrics[f"f{beta}"]["optimal_thresholds"].append(optimal_threshold)

        if max_fbetascore > fmetrics[f"f{beta}"]["best_score"]:
            fmetrics[f"f{beta}"]["best_score"] = max_fbetascore
            fmetrics[f"f{beta}"]["best_fold"] = fold

        mlflow.log_metric(f"f{beta}-score", max_fbetascore)
        mlflow.log_metric(f"optimal_threshold_f{beta}", optimal_threshold)

    return fmetrics


def evaluate_and_log_metrics(
    preds, target, metrics=None, beta=2, fold=None, learn=None, best_epoch=None
):
    if metrics is None:
        metrics = initialize_metrics()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)

    # Calculate PR curve, AP, F-scores
    precision, recall, thresholds = precision_recall_curve(target, preds)
    ap = average_precision_score(target, preds)

    # Update metrics
    for metric_name, value in [
        ("roc_auc", roc_auc),
        ("avg_precision", ap),
    ]:
        metrics[metric_name]["scores"].append(value)
        if value > metrics[metric_name]["best_score"]:
            metrics[metric_name]["best_score"] = value
            metrics[metric_name]["best_fold"] = fold
            if learn:
                learn.save(f"best_{metric_name}")
                metrics[metric_name]["best_model"] = f"best_{metric_name}"

    metrics["precision"]["scores"].append(precision)
    metrics["recall"]["scores"].append(recall)
    metrics["fpr"]["scores"].append(fpr)
    metrics["tpr"]["scores"].append(tpr)
    if best_epoch:
        metrics["meta"]["best_epoch"].append(best_epoch)

    # Log metrics for each fold
    for metric_name, value in [
        ("roc_auc", roc_auc),
        ("avg_precision", ap),
    ]:
        mlflow.log_metric(metric_name, value)

    latest_metrics = {
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
    }

    return metrics, latest_metrics
