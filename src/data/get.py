import pandas as pd
import numpy as np
from src.data.utils import cfg

from src.common.log_config import setup_logging, clear_log
import logging

setup_logging()
logger = logging.getLogger(__name__)


def get_concept(concept, cfg) -> dict:
    """get numerical concept from name"""

    drop_cols = cfg["drop_features"][concept]
    concept_dict = {}

    for agg_func in cfg["agg_func"][concept]:
        logger.info(f"Loading {concept}.agg_func: {agg_func}")
        df = pd.read_csv(f"data/interim/mapped/{concept}_{agg_func}.csv", index_col=0)
        # if features to drop, drop now.
        try:
            df = df[
                ~df.FEATURE.isin(
                    drop_cols
                    + [
                        np.nan,
                    ]
                )
            ]
        except:
            df = df[
                ~df.FEATURE.isin(
                    [
                        np.nan,
                    ]
                )
            ]
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        concept_dict[agg_func] = df
    return concept_dict
