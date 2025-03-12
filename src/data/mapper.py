import dask.dataframe as dd

from src.common.common_imports import *
from src.data.processors import merge_and_aggregate
from src.data.utils import get_bin_df
from src.data.filters import filter_vitals, collect_filter


def map_concept(cfg, concept: str, agg_func):
    # Load binning DataFrame
    bin_df = get_bin_df()
    logger.info(f"Prepared bin df, now preparing {concept}")

    # Load and filter concept
    concept_df = pd.read_pickle(f"data/interim/{concept}.pkl")
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)

    # Process each feature
    dfs = []
    logger.info(f"Processing each feature")

    for feat in concept_df.FEATURE.unique():
        logger.info(f"start {feat}")
        subset = concept_df[concept_df.FEATURE == feat]

        # Merge and aggregate
        logger.info(f"{feat} merging and aggregating")
        result_df = merge_and_aggregate(bin_df, subset, agg_func=agg_func)

        dfs.append(result_df)

    logger.info("Concatenating feature dfs")
    # concat dataframes into one long
    result_df = (
        pd.concat(dfs)
        .drop_duplicates()
        .sort_values(["PID", "bin_counter"])
        .reset_index(drop=True)
    )

    # remove bin placeholder/nan rows if data rows present
    grouped = result_df.groupby(["PID", "bin_counter"])

    # Function to filter rows
    def filter_rows(group):
        if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
            return group
        else:
            return group.dropna(subset=["FEATURE", "VALUE"])

    logger.info(f"Cleaning binned {concept} dataframe")
    # Applying the filter function to each group
    filtered_df = grouped.apply(filter_rows).reset_index(drop=True)
    output_path = f"data/interim/mapped/{concept}"
    logger.info(f"Saving file to {output_path}")
    filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
    filtered_df.to_csv(f"{output_path}_{agg_func}.csv")


def map_concept_dask(cfg, concept: str, agg_func):
    # load binning df
    bin_df = get_bin_df()
    # as dask
    bin_ddf = dd.from_pandas(bin_df, npartitions=cfg["npartitions"]["bin_df"])
    logger.info(f"Prepared bin df, now preparing {concept}")
    logger.debug(f"bin df len: {len(bin_df)}")

    # load and filter concept
    concept_df = pd.read_pickle(f"data/interim/{concept}.pkl")

    # get releveant filter func and apply
    filter_function = collect_filter(concept)
    concept_df = filter_function(concept_df)

    # split vitals by feature, merge aggregate and append feature specific df to list
    dfs = []
    logger.info(f"Processing each feature with partitions")

    for feat in concept_df.FEATURE.unique():
        logger.info(f"start {feat}")
        subset = concept_df[concept_df.FEATURE == feat]

        # Convert to Dask DataFrames
        try:
            npartitions = cfg["npartitions"][concept]
        except:
            npartitions = 5
        subset_ddf = dd.from_pandas(concept_df, npartitions=npartitions)
        # proces and append to list
        logger.info(f"{feat} merging and aggregating by {npartitions} partitions")

        # TODO: change this to be flexible instead of only numerical input feat
        # TODO: change merge_and_agg to perform all aggs same looping...
        result_ddf = merge_and_aggregate(bin_ddf, subset_ddf, agg_func=agg_func)
        result_df = result_ddf.compute()

        dfs.append(result_df)

    logger.info("Concatenating feature dfs")
    # concat dataframes into one long
    result_df = (
        pd.concat(dfs)
        .drop_duplicates()
        .sort_values(["PID", "bin_counter"])
        .reset_index(drop=True)
    )

    # remove bin placeholder/nan rows if data rows present
    grouped = result_df.groupby(["PID", "bin_counter"])

    # Function to filter rows
    def filter_rows(group):
        if group["FEATURE"].isna().all() and group["VALUE"].isna().all():
            return group
        else:
            return group.dropna(subset=["FEATURE", "VALUE"])

    logger.info(f"Cleaning binned {concept} dataframe")
    # Applying the filter function to each group
    filtered_df = grouped.apply(filter_rows).reset_index(drop=True)
    output_path = f"data/interim/mapped/{concept}"
    logger.info(f"Saving file to {output_path}")
    filtered_df.to_pickle(f"{output_path}_{agg_func}.pkl")
    filtered_df.to_csv(f"{output_path}_{agg_func}.csv")
