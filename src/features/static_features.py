import pandas as pd
import numpy as np

import subprocess

from src.data.utils import cfg, get_base_df
from src.common.log_config import setup_logging, clear_log
from src.utils import is_file_present, are_files_present
from src.utils import inches_to_cm, ounces_to_kg

import logging

setup_logging()
logger = logging.getLogger(__name__)


def add_to_base(base):

    base["DURATION"] = (base.end - base.start) / np.timedelta64(1, "D")

    base["AGE"] = (
        np.floor(
            (pd.to_datetime(base["start"]) - pd.to_datetime(base.DOB)).dt.days / 365.25
        )
    ).astype(int)

    base = add_height_weight(base)
    # Mortality
    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=30),
        "deceased_30d",
    ] = 1
    base["deceased_30d"] = base["deceased_30d"].fillna(0)

    base.loc[
        (pd.to_datetime(base.DOD) - pd.to_datetime(base.start))
        <= pd.Timedelta(days=90),
        "deceased_90d",
    ] = 1
    base["deceased_90d"] = base["deceased_90d"].fillna(0)
    # If trauma bay RH
    base["TB"] = 0
    base.loc[base.first_RH.notnull(), "TB"] = 1

    return base


def prepare_long_df(base):
    diag = pd.read_csv("data/raw/Diagnoser.csv")

    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])

    merged_df = base[["CPR_hash", "PID", "AGE", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )

    # Filtering rows where Noteret_dato is between start and end
    filtered_df = merged_df[
        (merged_df["Noteret_dato"] >= merged_df["start"] - pd.DateOffset(days=1))
        & (merged_df["Noteret_dato"] <= merged_df["end"] + pd.DateOffset(days=1))
    ]

    # Adjust Diagnosekode by removing the first and last character for ICD10 conversion
    filtered_df["Diagnosekode"] = filtered_df["Diagnosekode"].str.slice(1, -1)

    # Now, checking how many unique combinations are there
    logger.info(
        f"Unique CPR_hash-ServiceDate combinations in df1:{base.groupby('PID').ngroups}"
    )
    logger.info(
        f"Result after merging and filtering: {filtered_df.groupby('PID').ngroups}"
    )

    # Group by CPR_hash and apply a function to create new columns for each Diagnosekode
    def enumerate_diagnoses(group):
        diagnoses = group["Diagnosekode"].tolist()
        for i, diag in enumerate(diagnoses, start=1):
            group[f"ICD10_{i}"] = diag
        return group

    # Applying the function
    result_df = filtered_df.groupby("PID").apply(enumerate_diagnoses)

    # Dropping duplicates if necessary (since each row is expanded per group)
    result_df = result_df.drop_duplicates(subset="PID").reset_index(drop=True)

    result_df.to_csv("data/interim/ISS_ELIX/diagnoses_long.csv")


def prepare_elix_df(base):
    diag = pd.read_csv("data/raw/Diagnoser.csv")

    diag["Noteret_dato"] = pd.to_datetime(diag["Noteret_dato"])

    merged_df = base[["CPR_hash", "PID", "AGE", "start", "end"]].merge(
        diag, on="CPR_hash", how="left"
    )

    logger.info("Preparing Elixhauser Df")
    e_df = merged_df[
        (merged_df["Noteret_dato"] <= merged_df["start"] - pd.DateOffset(days=1))
        & (
            (merged_df["Løst_dato"] >= merged_df["start"] + pd.DateOffset(days=1))
            | (merged_df["Løst_dato"].isnull())
        )
    ]

    # Adjust Diagnosekode by removing the first and last character for ICD10 conversion
    e_df["Diagnosekode"] = e_df["Diagnosekode"].str.slice(1, -1)

    # Now, checking how many unique combinations are there
    logger.info(
        f"Unique CPR_hash-ServiceDate combinations in df1: {base.groupby('PID').ngroups}"
    )

    logger.info(f"Result after merging and filtering: {e_df.groupby('PID').ngroups}")
    e_df[["PID", "AGE", "Diagnosekode"]].to_csv("data/interim/ISS_ELIX/elix_df.csv")


def add_iss(base):
    """Add ISS and Elixhauser by R"""

    # Create long df if not there
    if is_file_present("data/interim/ISS_ELIX/diagnoses_long.csv"):
        logger.info("Long diagnose df dataframe found, continuing")
    else:
        logger.info("No long diagnose file, creating.")
        prepare_long_df(base)
    logger.info("Calling R script to create ISS df at data/interim/ISS_ELIX/iss_df.csv")
    subprocess.call("Rscript src/R/iss.r", shell=True)
    logger.info("R subprocess finished")


def add_elixhauser(base):
    if is_file_present("data/interim/ISS_ELIX/elix_df.csv"):
        logger.info("Elixhauser diagnose df dataframe found, continuing")
    else:
        logger.info("No Elixhauser diagnose file, creating.")
        prepare_elix_df(base)
    logger.info("Calling R script to create Elixhauser df at data/interim/ISS_ELIX/")
    subprocess.call("Rscript src/R/elixhauser.r", shell=True)
    logger.info("R subprocess finished")


def prepare_height_weight(base):
    vit_raw = pd.read_csv("data/raw/VitaleVaerdier.csv", index_col=0)
    hw_map = {"Højde": "HEIGHT", "Vægt": "WEIGHT"}
    vit_raw.rename(
        columns={
            "Værdi": "VALUE",
            "Vital_parametre": "FEATURE",
            "Registreringstidspunkt": "TIMESTAMP",
        },
        inplace=True,
    )

    vit_raw["FEATURE"] = vit_raw["FEATURE"].replace(to_replace=hw_map)
    vit_raw.loc[vit_raw.FEATURE == "HEIGHT", "VALUE"] = inches_to_cm(
        vit_raw[vit_raw.FEATURE == "HEIGHT"].VALUE.astype(float)
    )
    vit_raw.loc[vit_raw.FEATURE == "WEIGHT", "VALUE"] = ounces_to_kg(
        vit_raw[vit_raw.FEATURE == "WEIGHT"].VALUE.astype(float)
    )
    hw = vit_raw[(vit_raw.FEATURE.isin(list(set(hw_map.values()))))]
    hw = hw.merge(base[["PID", "CPR_hash", "start", "end"]], on="CPR_hash", how="left")
    hw["TIMESTAMP"] = pd.to_datetime(hw.TIMESTAMP)
    hw = hw[hw.TIMESTAMP <= hw.end]
    hw = hw.sort_values(["CPR_hash", "delta"]).drop_duplicates(
        subset=["CPR_hash", "FEATURE"], keep="first"
    )
    hw = hw[hw.delta.dt.days < 365 * 2]
    hw[["TIMESTAMP", "PID", "FEATURE", "VALUE"]].to_pickle(
        "data/interim/Height_Weight.pkl"
    )


def add_height_weight(base):
    hw = pd.read_pickle("data/interim/Height_Weight.pkl")
    hw_df = hw.sort_values("TIMESTAMP").drop_duplicates(
        subset=["PID", "FEATURE"], keep="first"
    )
    pivot_df = hw_df.pivot(
        index=["PID"], columns="FEATURE", values="VALUE"
    ).reset_index()
    base = base.merge(pivot_df, how="left", on="PID")

    return base
