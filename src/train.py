import gc
import json
import pandas as pd
import numpy as np

import torch

from sklearn.model_selection import StratifiedKFold
from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import (
    TSStandardize,
)
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy
from tsai.all import (
    TSTabFusionTransformer,
    LabelSmoothingCrossEntropyFlat,
    Learner,
    RocAucBinary,
    APScoreBinary,
)

from src.dataset.timeseries import TSDS
from src.data.utils import align_dataframes
from src.visualize import (
    plot_loss,
    plot_fold_evaluation,
)
from src.common.log_config import setup_logging, clear_log

from src.training.utils import initialize_metrics, evaluate_and_log_metrics
from src.training.utils import (
    initialize_fmetrics,
    evaluate_and_log_fmetrics,
    calculate_fmetrics_stats,
)
from src.training.utils import save_metrics
from src.training.tsai_custom import CSaveModel

import mlflow
from azureml.core import Workspace

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


import logging

setup_logging()
clear_log()
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="gridsearch.yaml")
def train(cfg: DictConfig):
    #   cfg = get_cfg()
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load data
        concepts = ["VitaleVaerdier", "Labsvar", "Medicin", "ITAOversigtsrapport"]
        # concepts= ['VitaleVaerdier']
        trainval = TSDS(cfg, filename="trainval_tab_simple", concepts=concepts)
        holdout = TSDS(cfg, filename="holdout_tab_simple", concepts=concepts)

        trainval.complete = pd.concat(trainval.concepts).fillna(0.0)
        holdout.complete = pd.concat(holdout.concepts).fillna(0.0)
        trainval.complete, holdout.complete = align_dataframes(
            trainval.complete, holdout.complete
        )
        # Setup
        exclude = ["deceased_30d"]
     
        cat_cols = OmegaConf.to_object(cfg["dataset"]["cat_cols"])
        ppj_cat_cols = OmegaConf.to_object(cfg["dataset"]["ppj_cat_cols"])
        
        cat_cols = cat_cols + ppj_cat_cols
        num_cols = [
            i
            for i in trainval.tab_df.columns
            if i not in cat_cols and i != cfg["target"] and i not in exclude
        ]
        logger.info(f"Categoricals: {cat_cols}\nNumericals: {num_cols}")

        mlflow.log_metric("Trainval N_PID:", len(trainval.base.PID.unique()))
        mlflow.log_metric("Holdout N_PID:", len(holdout.base.PID.unique()))

        # FOR CLASSES!
        tfms = [None, [Categorize()]]
        batch_tfms = TSStandardize(by_var=True)
        procs = [Categorify, FillMissing, Normalize]
        complete_tab_dls = get_tabular_dls(
            pd.concat([trainval.base, holdout.base]),
            procs=procs,
            cat_names=cat_cols.copy(),
            cont_names=num_cols.copy(),
            y_names=cfg["target"],
            splits=None,
            drop_last=False,
            shuffle=False,
        )
        classes = complete_tab_dls.classes

        # Training and validation dataset
        logger.info("Setting up X,y for training and validation")
        X, y = df2xy(
            trainval.complete,
            sample_col="PID",
            feat_col="FEATURE",
            data_cols=trainval.complete.columns[3:],
            target_col=cfg["target"],
        )
        y = y[:, 0].flatten()
        y = list(y)
        logger.info(f"Train/val X shape: {X.shape}")

        # TRAINING
        logger.info("Preparing training cycle")
        # Initialize vars to store results
        metrics = initialize_metrics()
        fmetrics = initialize_fmetrics()
        epoch_list = []

        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(
            n_splits=cfg["training"]["folds"], shuffle=True
        )  # , random_state=92) # not seeding to extract more info
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                # mlflow.autolog()
                logger.info(f"Fold {fold}")

                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()

                # Create splits
                ts_splits = (list(train_idx), list(val_idx))

                ts_dls = get_ts_dls(
                    X,
                    y,
                    splits=ts_splits,
                    tfms=tfms,
                    batch_tfms=batch_tfms,
                    bs=cfg["training"]["bs"],
                    drop_last=False,
                )

                tab_dls = get_tabular_dls(
                    trainval.tab_df,
                    procs=procs,
                    cat_names=cat_cols.copy(),
                    cont_names=num_cols.copy(),
                    y_names=cfg["target"],
                    splits=ts_splits,
                    bs=cfg["training"]["bs"],
                    drop_last=False,
                )

                mixed_dls = get_mixed_dls(ts_dls, tab_dls, bs=cfg["training"]["bs"])

                # Logging data and splits:
                trainval.tab_df.to_csv("models/tables/tab_df.csv")
                mlflow.log_artifact("models/tables/tab_df.csv")
                # Save splits for this fold
                fold_splits = {"train": train_idx.tolist(), "val": val_idx.tolist()}
                with open(f"fold_{fold}_splits.json", "w") as f:
                    json.dump(fold_splits, f)

                # Log the splits file as an artifact
                mlflow.log_artifact(f"fold_{fold}_splits.json")

                # classes = tab_dls.classes MOVED THIS!
                logger.debug(classes)
                model = TSTabFusionTransformer(
                    mixed_dls.vars,
                    2,
                    mixed_dls.len,
                    classes,
                    num_cols,
                    n_layers=cfg["model"]["n_layers"],
                    n_heads=cfg["model"]["n_heads"],
                    d_model=cfg["model"]["d_model"],
                    fc_dropout=cfg["model"]["fc_dropout"],
                    res_dropout=cfg["model"]["res_dropout"],
                    fc_mults=(cfg["model"]["fc_mults_1"], cfg["model"]["fc_mults_1"]),
                )

                logger.debug(f"numcols: {num_cols}")

                smcb = CSaveModel(
                    monitor="average_precision_score",
                    fname=f"model_fold{fold}",
                    every_epoch=False,
                    at_end=True,
                    with_opt=False,
                    reset_on_fit=False,
                    comp=None,
                )
                smcb_ref = smcb
                # loss_func = CrossEntropyLossFlat(weight = torch.FloatTensor([1, 15]))
                loss_func = LabelSmoothingCrossEntropyFlat()

                learn = Learner(
                    mixed_dls,
                    model,
                    loss_func=loss_func,
                    metrics=[RocAucBinary(), APScoreBinary()],
                    cbs=[smcb],
                )

                logger.info(learn.recorder.cbs)

                # Set learning rate from fastai lr-finder by hpm factor (most often reduced)
                if cfg["training"]["fixed_lr"]:
                    lr = cfg["training"]["lr"]
                else:
                    lr = learn.lr_find(show_plot=False).valley

                learn.fit_one_cycle(cfg["training"]["epochs"], lr)

                logger.debug(learn.recorder.cbs)

                best_epoch = smcb_ref.best_epoch
                logger.info(best_epoch)
                #     except:
                # best_epoch = learn.recorder.cbs[-1].best_epoch

                epoch_list.append(best_epoch)
                loss_fig, loss_df = plot_loss(learn, fold)
                loss_df.to_csv("models/tables/loss_df.csv")
                mlflow.log_figure(loss_fig, "loss.png")
                mlflow.log_artifact("models/tables/loss_df.csv")

                # Get predictions on validation set and save for reproduceabiltiy
                val_preds, val_targets = learn.get_preds(dl=learn.dls.valid)
                val_preds = (
                    val_preds[:, 1].cpu().numpy()
                )  # Move to CPU and convert to numpy
                val_targets = val_targets.cpu().numpy()

                preds_df = pd.DataFrame(
                    {"prediction": val_preds, "target": val_targets}
                )
                preds_df.to_csv(f"models/tables/pred_df_fold_{fold}")
                mlflow.log_artifact(f"models/tables/pred_df_fold_{fold}")

                metrics, latest_metrics = evaluate_and_log_metrics(
                    val_preds,
                    val_targets,
                    metrics,
                    fold=fold,
                    beta=cfg["evaluation"]["f_beta"],
                    best_epoch=best_epoch,
                )
                fmetrics = evaluate_and_log_fmetrics(
                    val_preds, val_targets, fmetrics, fold=fold, learn=learn
                )

                # Log best scores after each fold
                # for metric_name in ["roc_auc", "avg_precision", "fscore", "fbetascore"]:
                #    logger.info(f'Best {metric_name}: {metrics[metric_name]["best_score"]} (Fold {metrics[metric_name]["best_fold"]})')

                learn.save(f"fold_{fold}")

                # Clear model from GPU
                learn.model = learn.model.cpu()
                del learn
                gc.collect()
                torch.cuda.empty_cache()
        valid_eval_fig = plot_fold_evaluation(metrics, target=cfg["target"])
        mlflow.log_figure(valid_eval_fig, "val_eval.png")

        pd.DataFrame(fmetrics).to_csv("models/f_metrics.csv")
        pd.DataFrame(metrics).to_csv("models/metrics.csv")

        save_metrics(metrics, "models/metrics.json")
        mlflow.log_artifacts("models/metrics.json")

        fmetrics_stats = calculate_fmetrics_stats(fmetrics)

        # Add artifacts to mlflow
        artifact_paths = [
            "logging/app.log",
            "conf/gridsearch.yaml",
            "models/f_metrics.csv",
            "models/metrics.csv",
        ]
        for artifact in artifact_paths:
            mlflow.log_artifact(artifact)
        mlflow.log_metric("best_epoch_avg", np.average(epoch_list))

        return np.average(metrics["avg_precision"]["scores"])


if __name__ == "__main__":
    train()
