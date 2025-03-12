import os
# set workdir as project root for notebook imports.
compute_name = 'amil-gpu1'
os.chdir(f'/mnt/batch/tasks/shared/LS_root/mounts/clusters/{compute_name}/code/Users/andreas.skov.millarch/repos/DTTD')


import numpy as np

from fastai.tabular.all import *#TabularPandas, RandomSplitter, CrossEntropyLossFlat, tabular_learner
from src.utils import clear_mem

import pandas as pd 
import numpy as np 

from tsai.data.core import get_ts_dls
from tsai.data.preprocessing import TSStandardize
from tsai.data.tabular import get_tabular_dls
from tsai.data.mixed import get_mixed_dls
from tsai.data.preparation import df2xy

from fastai.data.transforms import Categorize
from fastai.tabular.core import Categorify, FillMissing, Normalize
from fastai.callback.core import TrainEvalCallback


from src.visualize import plot_loss

from src.data.utils import get_cfg
from src.training.utils import initialize_metrics
from src.training.utils import initialize_fmetrics 
from src.common.log_config import setup_logging, clear_log
from src.evaluation import calculate_roc_auc_ci, evaluate, calculate_average_precision_ci
from src.dataset.timeseries import TSDS
from src.data.utils import align_dataframes
from tsai.all import TSTabFusionTransformer#,SaveModel
from tsai.all import get_tabular_dls

from src.evaluation import get_internal_validation_thresholds

import logging 

setup_logging()
clear_log()
logger = logging.getLogger(__name__)

cfg = get_cfg()

# Create a custom callback to skip validation
class SkipValidationCallback(TrainEvalCallback):
    def before_validate(self): 
        raise CancelValidException()

def plot_loss(learn):
    skip_start = 0
    plt.plot(list(range(skip_start, len(learn.recorder.losses))), learn.recorder.losses[skip_start:], label='train')
    idx = (np.array(learn.recorder.iters)<skip_start).sum()
    plt.legend()
    plt.savefig('models/loss.png', dpi=1200)    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    concepts= ['VitaleVaerdier', 'Labsvar', 'Medicin','ITAOversigtsrapport']#, 'Procedurer']
    #concepts= ['VitaleVaerdier']
    trainval = TSDS(cfg, filename='trainval_tab_simple', concepts=concepts)
    holdout = TSDS(cfg, filename='holdout_tab_simple', concepts=concepts)

    trainval.complete = pd.concat(trainval.concepts).fillna(0.0)
    holdout.complete = pd.concat(holdout.concepts).fillna(0.0)
    trainval.complete, holdout.complete = align_dataframes(trainval.complete, holdout.complete)

    # Setup
    exclude = ['deceased_30d']

    #cat_cols = ['SEX']
    #cat_cols=cfg["dataset"]["cat_cols"]
    cat_cols = ['SEX', 'LVL1TC']
    #ppj_cat_cols = cfg["dataset"]["ppj_cat_cols"]
    ppj_cat_cols = ["A","B","C","D"]

    cat_cols = cat_cols+ppj_cat_cols
    num_cols = [i for i in trainval.tab_df.columns if i not in cat_cols and i != cfg["target"] and i not in exclude]
    logger.info(f'Categoricals: {cat_cols}\nNumericals: {num_cols}')

    # HOLDOUT 

    # Note, no batch shuffle on dataloder
    logger.info('Preparing holdout dataloader')
    tfms = [None, [Categorize()]]
    batch_tfms = TSStandardize(by_var=True)
    procs = [Categorify, FillMissing, Normalize]
    # Holdout set
    tX, ty = df2xy(holdout.complete, 
                sample_col='PID', 
                feat_col='FEATURE', 
                data_cols=holdout.complete.columns[3:], 
                target_col=holdout.target)
    ty = ty[:, 0].flatten()
    ty = list(ty)
    logger.info(f'Holdout X shape: {tX.shape}')
    test_ts_dls = get_ts_dls(tX, ty, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=cfg["training"]["bs"], drop_last=False,  shuffle=False)

    test_tab_dls = get_tabular_dls(holdout.base, procs=procs, 
                                cat_names=cat_cols.copy(), 
                                cont_names=num_cols.copy(), 
                                y_names=cfg["target"], 
                                splits=None, 
                                drop_last=False,
                                shuffle=False)


    holdout_mixed_dls = get_mixed_dls(test_ts_dls, test_tab_dls, bs=cfg["training"]["bs"], shuffle_valid=False)

    # FOR CLASSES!
    complete_tab_dls = get_tabular_dls(pd.concat([trainval.base, holdout.base]), procs=procs, 
                            cat_names=cat_cols.copy(), 
                            cont_names=num_cols.copy(), 
                            y_names=cfg["target"], 
                            splits=None, 
                            drop_last=False,
                            shuffle=False)
    classes = complete_tab_dls.classes

    # Training and validation dataset
    logger.info("Setting up X,y for training and validation")
    X, y = df2xy(trainval.complete, 
            sample_col='PID', 
            feat_col='FEATURE', 
            data_cols=trainval.complete.columns[3:], 
            target_col=cfg["target"])
    y = y[:, 0].flatten()
    y = list(y)
    logger.info(f'Train/val X shape: {X.shape}')

    # TRAINING
    logger.info('Preparing training cycle')
    # Initialize vars to store results
    metrics = initialize_metrics()
    fmetrics = initialize_fmetrics()

    ts_dls = get_ts_dls(X, y, splits=None, tfms=tfms, batch_tfms=batch_tfms, bs=cfg["training"]["bs"], drop_last=False)

    tab_dls = get_tabular_dls(trainval.tab_df, procs=procs, 
                            cat_names=cat_cols.copy(), 
                            cont_names=num_cols.copy(), 
                            y_names=cfg["target"], 
                            splits=None, bs=cfg["training"]["bs"],
                            drop_last=False)

    mixed_dls = get_mixed_dls(ts_dls, tab_dls, 
                            bs=cfg["training"]["bs"])
    
    model = TSTabFusionTransformer(c_in = mixed_dls.vars, c_out = 2,
                                      seq_len =  mixed_dls.len, classes=classes, cont_names= num_cols, 
                                       n_layers = 8,
                                       n_heads = 8,
                                       d_model=64,
                                       fc_dropout=0.75, 
                                       res_dropout=0.22,
                                      fc_mults = (0.3, 0.1),
                              )

    logger.debug(f'numcols: {num_cols}')

    loss_func = LabelSmoothingCrossEntropyFlat()

    learn = Learner(mixed_dls, model,
                    loss_func=loss_func,
                    metrics=None,
                    cbs=[SkipValidationCallback()]
                )
    logger.debug(mixed_dls.vars, 2, f"seq_len={mixed_dls.len}",f"n_cat ={len(classes.keys())}", f"n_cont={len(num_cols)}" )

    lr = 0.0004786300996784121

    learn.fit_one_cycle(20, lr)    
    plot_loss(learn)

    learn.save('full_model')
    # Load saved model into learner for evaluation
    # Necessary due to the modified training proces without validation set before
    clear_mem()

    model = TSTabFusionTransformer(mixed_dls.vars, 2,
                                       mixed_dls.len, classes, num_cols, 
                                       n_layers = 8,
                                       n_heads = 8,
                                       d_model=64,
                                       fc_dropout=0.75, 
                                       res_dropout=0.22,
                                      fc_mults = (0.3, 0.1),
                              )

    logger.debug(f'numcols: {num_cols}')

    loss_func = LabelSmoothingCrossEntropyFlat()

    learn = Learner(mixed_dls, model,
                    loss_func=loss_func,
                    metrics=None,
                )
    learn = learn.load( 'full_model')

    beta=5
    thresholds = get_internal_validation_thresholds(beta=beta)
    thresholds = [t for t in thresholds if t <0.5]
    threshold = np.average(thresholds)

    evaluate(holdout_mixed_dls, learn, threshold)

    preds, target = learn.get_preds(dl=holdout_mixed_dls.train)
    preds = preds[:,1].cpu().numpy()
    target = target.cpu().numpy()

    auc, auc_ci_lower, auc_ci_upper = calculate_roc_auc_ci(target, preds)
    ap, ap_ci_lower, ap_ci_upper = calculate_average_precision_ci(target, preds)

    logger.info(f"ROC AUC: {auc:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})")
    logger.info(f"Average Precision: {ap:.3f} (95% CI: {ap_ci_lower:.3f}-{ap_ci_upper:.3f})")

if __name__ == '__main__':
    main()