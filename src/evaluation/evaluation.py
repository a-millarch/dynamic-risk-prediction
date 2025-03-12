import numpy as np
import matplotlib.pyplot as plt

from azureml.core import Experiment, Run, Workspace
import mlflow
from src.training.utils import get_threshold
import numpy as np 
import ast
import os

from src.visualize import plot_evaluation, create_calibration_plot, evaluate_detection_rate, plot_hist_kde
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from scipy import stats
from sklearn.metrics import roc_auc_score

def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance using DeLong's method.
    
    Args:
    ground_truth (array-like): True binary labels
    predictions (array-like): Target scores (probability estimates)
    
    Returns:
    float: Variance of ROC AUC
    """
    order = np.argsort(predictions)
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    
    n_pos = np.sum(ground_truth)
    n_neg = len(ground_truth) - n_pos
    
    pos_ranks = np.where(ground_truth == 1)[0] + 1
    neg_ranks = np.where(ground_truth == 0)[0] + 1
    
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    v01 = (auc / (2 - auc) - auc ** 2) / (n_neg)
    v10 = (2 * auc ** 2 / (1 + auc) - auc ** 2) / (n_pos)
    
    return v01 + v10

def calculate_roc_auc_ci(y_true, y_pred, alpha=0.95):
    """
    Calculate ROC AUC score and its confidence interval using DeLong method.
    
    Args:
    y_true (array-like): True binary labels
    y_pred (array-like): Target scores (probability estimates)
    alpha (float): Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    tuple: (AUC score, lower bound of CI, upper bound of CI)
    """
    auc = roc_auc_score(y_true, y_pred)
    auc_var = delong_roc_variance(y_true, y_pred)
    
    auc_std = np.sqrt(auc_var)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci[ci > 1] = 1
    ci[ci < 0] = 0
    return auc, ci[0], ci[1]


def calculate_average_precision_ci(y_true, y_pred, alpha=0.95, n_bootstraps=1000):
    """
    Calculate average precision score and its confidence interval using bootstrapping.
    
    Args:
    y_true (array-like): True binary labels
    y_pred (array-like): Target scores (probability estimates)
    alpha (float): Confidence level (default: 0.95 for 95% CI)
    n_bootstraps (int): Number of bootstrap samples
    
    Returns:
    tuple: (Average precision score, lower bound of CI, upper bound of CI)
    """
    ap = average_precision_score(y_true, y_pred)
    
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = average_precision_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    ci_lower = sorted_scores[int((1.0-alpha)/2 * len(sorted_scores))]
    ci_upper = sorted_scores[int((1.0+alpha)/2 * len(sorted_scores))]
    
    return ap, ci_lower, ci_upper

def is_float_in_range(value, start, end, step=0.1):
    return start <= round(float(value), 1) <= end and round(float(value) % step, 1) == 0

def dl_file_from_run(experiment, run_id, filename, output_prefix="", output_path= 'models/study/eval/'):
    run = Run(experiment, run_id)
    # find model in run, download to temporary directory
    file_path = ["./"+i for i in run.get_file_names() if filename in i]
    file_names = list(map(lambda x: x.split('/')[-1], file_path))
    if len(file_names) == 1:
        file_name = file_names[0]
        tmp_path = output_path+output_prefix+file_name
        run.download_file(file_path[0], tmp_path)
    elif len(file_names) == 1:
        print("multiple files found")
        for file_name, fp  in zip(file_names,file_path):
            tmp_path = output_path+output_prefix+file_name
            run.download_file(fp, tmp_path)  
    else:
        print("no file found")
    
def get_internal_validation_thresholds(beta=2):

    experiment_name = 'NEJM'
    output_prefix=""
    print(f"str{output_prefix}")


    # Get the Azure ML workspace
    workspace = Workspace.from_config()

    # Set the MLflow tracking URI to point to your Azure ML workspace
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    workspace = Workspace.from_config()
    experiment = Experiment(workspace=workspace, name=experiment_name)
    runs = experiment.get_runs()

    keep_run_ids = []
    for idx, run in enumerate(runs):
        r = mlflow.get_run(run.id)
       # print(r)
        if r.data.params["model"]:
            model_params = ast.literal_eval(r.data.params["model"])
            if (is_float_in_range(model_params.get('res_dropout', 0), 0.2, 0.3) and 
                is_float_in_range(model_params.get('fc_dropout', 0), 0.7, 0.8) and
                model_params.get('n_layers') == 8 and
                model_params.get('n_heads') == 8 and
                model_params.get('d_model') == 64):
                    keep_run_ids.append(run.id)
                    print(f"kept run.id {run.id}")
        
    runs = experiment.get_runs()
    for idx,run in enumerate(runs):
        if run.id in keep_run_ids:
            for child in run.get_children():
                dl_file_from_run(experiment, child.id, "pred_df",output_prefix=f"preds/run_{idx}_")

    folder_path = 'models/study/eval/preds'
    thresholds= []
    for filename in os.listdir(folder_path):
        if filename.startswith('run'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, index_col=0)
            threshold = get_threshold(df["target"], df["prediction"], method="Fbeta-score", beta=beta)
            thresholds.append(threshold)
    return thresholds


def calculate_net_benefit(y_true, y_pred, threshold):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure y_pred is probability (between 0 and 1)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 1]
    y_pred = np.clip(y_pred, 0, 1)
    
    tp = np.sum((y_pred >= threshold) & (y_true == 1))
    fp = np.sum((y_pred >= threshold) & (y_true == 0))
    n = len(y_true)
    
    # Avoid division by zero
    if threshold == 1:
        return 0
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

def calculate_always_act(y_true, threshold):
    prevalence = np.mean(y_true)
    # Avoid division by zero
    if threshold == 1:
        return 0
    return prevalence - (1 - prevalence) * (threshold / (1 - threshold))

def plot_net_benefit(y_true, y_pred):
    thresholds = np.linspace(0, 1, 100)

    net_benefits = [calculate_net_benefit(y_true, y_pred, t) for t in thresholds]
    always_act = [calculate_always_act(y_true, t) for t in thresholds]
    never_act = [0] * len(thresholds)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, label='Model')
    plt.plot(thresholds, always_act, label='Always Act')
    plt.plot(thresholds, never_act, label='Never Act')
    
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Net Benefit Curve')
    plt.legend()
    plt.grid(True)

    plt.ylim(-0.05, max(max(net_benefits), max(always_act)) + 0.05)
    
    plt.show()


def evaluate(holdout_mixed_dls, learn , threshold):
    # Get predictions on validation set
    preds, target = learn.get_preds(dl=holdout_mixed_dls.train)
    preds = preds[:,1].cpu().numpy()
    target = target.cpu().numpy()

    fig = plot_evaluation(preds, target,'90-day mortality')

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(target, preds)
    roc_auc = auc(fpr, tpr)

    
    # Calculate PR curve and AP
    precision, recall, _ = precision_recall_curve(target, preds)

    ap = average_precision_score(target, preds)

    # Calibration plot
    fig = create_calibration_plot(target, preds, n_bins=4)
    
    # KDE plot
    preds_df = pd.DataFrame()
    preds_df['preds'] = preds.astype(float)
    preds_df['target'] = target.astype(float)
    #fig =plot_box_kde(preds_df, 'target', 'preds')
    fig =plot_hist_kde(preds_df, 'target', 'preds')

    
    # CM: Detection rate
    fig = evaluate_detection_rate(y_preds=preds, y_true=target, threshold=threshold)
        
    plot_net_benefit(target, preds)

