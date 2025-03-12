import os
import pyarrow.parquet as pq
from azureml.core import Dataset

import logging

from src.utils import is_file_present, are_files_present
from src.data.utils import cfg, get_base_df
from src.utils import is_file_present, are_files_present
from src.common.log_config import setup_logging
from src.data.downloader import download_to_local


setup_logging()
logger = logging.getLogger(__name__)


def collect_subsets(cfg):
    # First, load small files using population filter function
    for filename in cfg["default_load_filenames"]:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f"{filename} found in raw")
        else:
            population_filter_parquet(filename)

    # For larger files, we need to first download parquet to local first
    if are_files_present(
        "dl", ["CPMI_" + i for i in cfg["large_load_filenames"]], extension=".parquet"
    ):
        logger.info("parquet files found locally, continue")
    else:
        logger.info("missing local parquet files, downloading")
        download_to_local(cfg["large_load_filenames"])

    # Now chunk filter to only population
    for filename in cfg["large_load_filenames"]:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f"{filename} found in raw")
        else:
            logger.info(f"Processing {filename}")
            chunk_filter_parquet(filename)


def collect_procedures(cfg=cfg):
    path = f'{cfg["raw_file_path"]}CPMI_Procedurer.parquet'
    df_procedure = Dataset.Tabular.from_parquet_files(path=path)
    dtr_procedure = df_procedure.to_pandas_dataframe()
    traumepatienter = dtr_procedure[dtr_procedure["ProcedureCode"] == "BWST1F"][
        ["CPR_hash", "ServiceDate"]
    ]
    traumepatienter.to_csv("data/raw/Procedurer_population.csv")


def chunk_filter_parquet(filename, base=None, chunk_size=4000000):
    if base is None:
        base = get_base_df()
        logger.info("Loaded base df")

    poplist = base["CPR_hash"].unique()
    # Specify the path to your Parquet file
    file_path = f"dl/CPMI_{filename}.parquet"
    output_path = f"data/raw/{filename}.csv"

    # Open the Parquet file
    parquet_file = pq.ParquetFile(file_path)

    # Define the chunk size
    # Adjust this as needed
    chunk_n = 0
    num_chunks = parquet_file.metadata.num_rows / chunk_size  # .round(0)
    logger.info(f">Initiating {num_chunks} chunks")
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk_n = chunk_n + 1
        print(f">>{chunk_n} of {num_chunks}chunks", end="\r")
        chunk_df = batch.to_pandas()
        chunk_df = chunk_df[chunk_df.CPR_hash.isin(poplist)]
        chunk_df.to_csv(output_path, mode="a", header=not os.path.exists(output_path))
    logger.info(f"Finished, saved file at: {output_path}")


def population_filter_parquet(filename, cfg=cfg, base=None):
    if base is None:
        base = get_base_df()

    logger.info(f"Collecting and filtering {filename}")
    path = f'{cfg["raw_file_path"]}CPMI_{filename}.parquet'
    ds = Dataset.Tabular.from_parquet_files(path=path)
    df = ds.to_pandas_dataframe()
    df = df[df.CPR_hash.isin(base.CPR_hash)]
    logger.info(f"loaded {len(df)} rows. Saving file.")

    df.to_csv(f"data/raw/{filename}.csv")


def population_bestordid_filter_parquet(filename, filter_cut_off=500000000):
    """Load population, load file, filter by BestOrd_ID, filter by pop, save (append save second load)"""
    if filename == "Medicin":
        filter_cut_off = 50000000

    pop = get_base_df()
    output_path = f"data/raw/{filename}.csv"
    logger.info(f"Collecting and filtering {filename}")
    ds = Dataset.Tabular.from_parquet_files(
        path=f'{cfg["raw_file_path"]}CPMI_{filename}.parquet'
    )
    df = ds.filter(ds["BestOrd_ID"] < filter_cut_off).to_pandas_dataframe()
    df[df.CPR_hash.isin(pop.CPR_hash)].to_csv(output_path)
    logger.info(f"loaded {len(df)} rows. Saving file.")

    df = ds.filter(ds["BestOrd_ID"] > filter_cut_off).to_pandas_dataframe()
    logger.info(f"loaded {len(df)} rows. Appending to saved file.")
    df[df.CPR_hash.isin(pop.CPR_hash)].to_csv(
        output_path, mode="a", header=not os.path.exists(output_path)
    )


if __name__ == "__main__":
    collect_subsets(cfg)
