import os
import pandas as pd
import logging

from src.data.creators import create_base_df, create_bin_df
from src.data.collectors import collect_subsets
from src.data.filters import filter_subsets_inhospital, filter_base
from src.data.mapper import map_vitals, map_concept
from src.dataset.tabular import TabDS
from src.data.utils import cfg, get_base_df
from src.utils import is_file_present
from src.common.log_config import setup_logging, clear_log

setup_logging()
logger = logging.getLogger(__name__)


def check_meta_file():
    if is_file_present("data/interim/base_meta.csv"):
        pass
    else:
        logger.info("> Creating base meta df")
        base = get_base_df()
        # remove by notes criteria
        base = filter_base(base)
        base = base[["PID", "ServiceDate", "start", "end", "TRAUMATEXT"]]
        base.to_csv("data/interim/base_meta.csv")
        logger.info(">> Finished")


def check_splits():
    #'data/processed/valdf.csv'
    pass  # TODO: create split in holdout _ OR! Is it done by tabds?


def check_analysis_df(overwrite=False):
    if (
        is_file_present("data/processed/trainval_tab_simple.csv")
        and is_file_present("data/processed/holdout_tab_simple.csv")
        and overwrite is False
    ):
        pass
    else:
        logger.info("> Creating tabular analysis dataframes")
        ds = TabDS(cfg, default_mode=True, exclude="study")
        ds.get_analysis_df(target=cfg["target"])
        logger.info(">> Finished")


def initialize(cfg):
    check_meta_file()
    check_splits()
    check_analysis_df(overwrite=cfg["dataset"]["overwrite"])


if __name__ == "__main__":
    initialize(cfg)
