from src.common.common_imports import *
import dask.dataframe as dd


# Function to merge and aggregate
def merge_and_aggregate_dask(bin_ddf, subset_ddf, agg_func="mean"):
    """
    Requires value to be numerical
    """
    # Ensure datatypes for memory efficiency
    bin_ddf["PID"] = bin_ddf["PID"].astype("int32")
    subset_ddf["PID"] = subset_ddf["PID"].astype("int32")
    subset_ddf["VALUE"] = subset_ddf["VALUE"].astype("float")

    # Merge on PID
    merged_ddf = dd.merge(bin_ddf, subset_ddf, on="PID", how="left")

    # Filter based on timestamp conditions
    filtered_ddf = merged_ddf[
        (merged_ddf["TIMESTAMP"] >= merged_ddf["bin_start"])
        & (merged_ddf["TIMESTAMP"] <= merged_ddf["bin_end"])
    ]

    # Aggregate the values
    aggregation = {
        "first": "first",
        "mean": "mean",
        "max": "max",
        "min": "min",
        "std": "std",
        "sum": "sum",
        "count": "count",
    }
    agg_function = aggregation.get(agg_func, "_")

    # Perform the aggregation
    aggregated_ddf = (
        filtered_ddf.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
        .agg({"VALUE": agg_function})
        .reset_index()
    )

    # Merge the result back to bin_ddf to maintain all rows
    result_ddf = dd.merge(
        bin_ddf,
        aggregated_ddf,
        on=[
            "PID",
            "bin_counter",
            "bin_start",
            "bin_end",
        ],
        how="left",
    )

    return result_ddf


def merge_and_aggregate(bin_df, subset_df, agg_func="mean"):
    # Ensure datatypes for memory efficiency
    bin_df["PID"] = bin_df["PID"].astype("int32")
    subset_df["PID"] = subset_df["PID"].astype("int32")
    subset_df["VALUE"] = subset_df["VALUE"].astype("float")

    # Merge on PID
    merged_df = pd.merge(bin_df, subset_df, on="PID", how="left")

    # Filter based on timestamp conditions
    filtered_df = merged_df[
        (merged_df["TIMESTAMP"] >= merged_df["bin_start"])
        & (merged_df["TIMESTAMP"] <= merged_df["bin_end"])
    ]

    # Aggregate the values
    aggregation = {
        "first": "first",
        "mean": "mean",
        "max": "max",
        "min": "min",
        "std": "std",
        "sum": "sum",
        "count": "count",
    }
    agg_function = aggregation.get(agg_func, "_")

    aggregated_df = (
        filtered_df.groupby(["PID", "bin_counter", "bin_start", "bin_end", "FEATURE"])
        .agg({"VALUE": agg_function})
        .reset_index()
    )

    # Merge the result back to bin_df to maintain all rows
    result_df = pd.merge(
        bin_df,
        aggregated_df,
        on=["PID", "bin_counter", "bin_start", "bin_end"],
        how="left",
    )

    return result_df
