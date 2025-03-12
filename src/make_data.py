import os
import pandas as pd
import logging

from src.data.creators import create_base_df, create_bin_df
from src.data.collectors import collect_subsets
from src.data.filters import filter_subsets_inhospital
from src.data.mapper import map_vitals, map_concept

from src.data.utils import cfg, get_base_df
from src.utils import is_file_present, are_files_present
from src.common.log_config import setup_logging, clear_log
from src.initialize import initialize

setup_logging()
logger = logging.getLogger(__name__)


def load_or_create_data(cfg, reset=False):
    # TODO: add arguments, to update complete or check for files and add new

    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]

    if is_file_present(cfg["base_df_path"]) and reset == False:
        logger.info("Base dataframe found, continuing")
    else:
        logger.info("No base file, creating.")
        create_base_df(cfg)

    if (
        are_files_present("data/raw", subsets_filenames, extension=".csv")
        and reset == False
    ):
        logger.info("All subsets found, continuing")
    else:
        logger.info("Subsets missing, collecting missing")
        collect_subsets(cfg)

    if (
        are_files_present("data/interim", subsets_filenames, extension=".pkl")
        and reset == False
    ):
        logger.info("Interim subsets found, continuing")
    else:
        logger.info("Filtering subsets")
        filter_subsets_inhospital(cfg)

    if is_file_present(cfg["bin_df_path"]) and reset == False:
        logger.info("Bin df found, continuing")
    else:
        logger.info("Creating bin dataframe")
        create_bin_df(cfg)


def map_vitals_legacy(cfg):
    for agg_func in cfg["agg_func"]["VitaleVaerdier"]:
        logger.info(f"Binning and mapping vitals with agg_func: {agg_func}")
        map_vitals(cfg, agg_func)


def map_data(cfg):

    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                logger.info(f"Binning and mapping {concept} with agg_func: {agg_func}")
                map_concept(cfg, concept, agg_func)


def map_data_dev(cfg):
    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        missing_agg_func_files = []
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                missing_agg_func_files.append(agg_func)

        if len(missing_agg_func_files) > 0:
            logger.info(
                f"Binning and mapping {concept} with agg_func: {missing_agg_func_files}"
            )
            map_concept(cfg, concept, missing_agg_func_files)
        else:
            logger.info(f"All mapped files found for {concept}")


def make_data(cfg):
    load_or_create_data(cfg)
    map_data(cfg)


if __name__ == "__main__":
    make_data(cfg)
    initialize(cfg)
