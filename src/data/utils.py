import pandas as pd
import numpy as np
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split


def get_cfg(cfg_path="conf/defaults.yaml"):
    with open(cfg_path) as file:
        return yaml.safe_load(file)


cfg = get_cfg()


def get_base_df(base_df_path=cfg["base_df_path"]):
    return pd.read_pickle(base_df_path)


def get_bin_df(bin_df_path=cfg["bin_df_path"]):
    return pd.read_pickle(bin_df_path)


def align_dataframes(df_a, df_b, fill_value=0.0):
    """
    Aligns two dataframes by adding missing columns to both and ensuring the same column order.
    Non-numeric column names come first, followed by numeric column names sorted in ascending order.

    Args:
    df_a (pd.DataFrame): First dataframe
    df_b (pd.DataFrame): Second dataframe
    fill_value (float): Value to fill in new columns (default: 0.0)

    Returns:
    tuple: (aligned_df_a, aligned_df_b)
    """

    # Get the union of all columns
    all_columns = set(df_a.columns) | set(df_b.columns)

    # Identify missing columns in each dataframe
    missing_in_a = all_columns - set(df_a.columns)
    missing_in_b = all_columns - set(df_b.columns)

    # Add missing columns to df_a
    for col in missing_in_a:
        df_a[col] = fill_value

    # Add missing columns to df_b
    for col in missing_in_b:
        df_b[col] = fill_value

    # Separate columns into non-numeric and numeric
    non_numeric_columns = [col for col in all_columns if not col.isdigit()]
    numeric_columns = sorted([col for col in all_columns if col.isdigit()], key=int)

    # Combine the columns with non-numeric first, followed by numeric
    sorted_columns = non_numeric_columns + numeric_columns

    # Reorder columns in both dataframes
    df_a_aligned = df_a.reindex(columns=sorted_columns)
    df_b_aligned = df_b.reindex(columns=sorted_columns)

    return df_a_aligned, df_b_aligned


def create_enumerated_id(df, string_col, datetime_col):
    # Combine the string and datetime columns to create a unique identifier
    df["unique_id"] = df[string_col].astype(str) + df[datetime_col].astype(str)

    # Drop duplicates to ensure each unique combination is only considered once
    df_unique = df.drop_duplicates(subset=["unique_id"]).copy()

    # Create an enumerated ID column
    df_unique["PID"] = range(1, len(df_unique) + 1)

    # Merge the new ID column back into the original DataFrame
    df = df.merge(df_unique[["unique_id", "PID"]], on="unique_id", how="left")

    # Drop the unique_id column as it's no longer needed
    df.drop(columns=["unique_id"], inplace=True)

    return df


def mark_keywords_in_df(
    df,
    text_column,
    keywords,
    timestamp=None,
    base_timestamp=None,
    t_delta=12,
    new_column="keyword_present",
):
    """
    Adds a new column to the DataFrame marking whether any of the keywords are present in the text column
    and checks if two datetime columns are within 12 hours apart.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text and datetime data.
    text_column (str): The name of the column with natural language text.
    keywords (list): A list of keywords to search for.
    datetime_col1 (str): The first datetime column.
    datetime_col2 (str): The second datetime column.
    new_column (str): The name of the new column to be added. Defaults to 'keyword_present'.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column marking keyword presence and datetime condition.
    """
    # Join the list of keywords into a regex pattern
    keyword_pattern = "|".join(keywords)

    # Check if keywords are present in the text column
    df[new_column] = df[text_column].str.contains(keyword_pattern, case=False, na=False)

    if timestamp and base_timestamp:
        # Check if the two datetime columns are within 12 hours of each other
        df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
        df[base_timestamp] = pd.to_datetime(df[base_timestamp], errors="coerce")

        df[f"within_{t_delta}_hours"] = (
            df[timestamp] - df[base_timestamp]
        ) <= pd.Timedelta(hours=t_delta)
    return df


def expand_datetime_rows(df):
    """WIP for medicine"""
    # Convert start and end columns to datetime
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Reduce dimensions
    df = (
        df[["PID", "TIMESTAMP", "FEATURE", "VALUE", "start", "end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f" df shape after reducing dimensions: {df.shape}")
    # Now, we want a row per minute from administration to seponation

    expanded_rows = []
    for _, row in df.iterrows():
        if pd.notnull(row["end"]):
            # Create a date range for each minute between start and end
            date_range = pd.date_range(start=row["start"], end=row["end"], freq="min")

        else:
            # If end is NaN, create a single-element range with just the start time
            date_range = pd.DatetimeIndex([row["start"]])

        # Create a new DataFrame for the current row's expanded data
        expanded_data = pd.DataFrame(
            {
                "PID": row["PID"],
                "TIMESTAMP": date_range,
                "VALUE": row["VALUE"],
                "FEATURE": row["FEATURE"],
            }
        )

        # Append to the list
        expanded_rows.append(expanded_data)

    # Concatenate all expanded DataFrames into a single DataFrame
    final_df = pd.concat(expanded_rows, ignore_index=True)
    return final_df.reset_index(drop=True)


def stratified_split_dataframe(df, target_column, test_size=0.2, random_state=None):
    """
    Splits a DataFrame into training and validation sets while retaining
    the same proportion of positive outcomes in the target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to split.
    - target_column (str): The name of the target column in the DataFrame.
    - test_size (float): The proportion of the dataset to include in the validation set.
    - random_state (int, optional): Controls the randomness of the split.

    Returns:
    - train_df (pd.DataFrame): The training dataset.
    - val_df (pd.DataFrame): The validation dataset.
    """

    # Define features and target
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target variable

    # Perform stratified split
    _, val_indices = train_test_split(
        df.index, test_size=test_size, stratify=y, random_state=random_state
    )

    # Create a HOLDOUT column
    df["HOLDOUT"] = False
    df.loc[val_indices, "HOLDOUT"] = True

    return df


# Transformation tools


def get_delta_time(base_df, input_df, base_df_date, input_df_date, mode="days"):
    """
    Returns input dataframe with a column containing time difference between the specified datetime variable and the base dataframe datetime variable.
    Use mode parameter to calculate difference in either days, hours or minutes.

    Parameters
    ----------
    base_df: pandas dataframe
        dataframe containing baseline datetime variable (t0)
        e.g. start of surgery

    input_df: pandas dataframe
        dataframe containing datetime variable (tx) for the event
        e.g. datetime of diagnosis

    base_df_date: str
        name of column for base_df datetimevariable.

    input_df_date: str
        name of column for inpu_df datetimevariable.

    mode: str
        'days', 'hours' or 'minutes'
    """

    input_df = input_df[input_df[input_df_date].notnull()].copy(deep=True)
    input_df[input_df_date] = pd.to_datetime(input_df[input_df_date])

    # input_df.loc[:, input_df_date] = pd.to_datetime(input_df[input_df_date], errors = 'coerce')
    base_df[base_df_date] = pd.to_datetime(base_df[base_df_date])

    df = input_df.merge(base_df[["CPR_hash", base_df_date]], on="CPR_hash", how="left")

    if mode.lower() in ["days", "d"]:
        df.loc[:, "delta_days"] = (df[input_df_date] - df[base_df_date]).dt.days

    elif mode.lower() in ["hours", "h"]:
        df.loc[:, "delta_hours"] = (df[input_df_date] - df[base_df_date]).dt.hour

    elif mode.lower() in ["minutes", "m"]:
        df.loc[:, "delta_minutes"] = (df[input_df_date] - df[base_df_date]).astype(
            "timedelta64[m]"
        )

    else:
        print("wrong mode. Use days, hours or minutes")

    return df


def remove_solved(df, df_date_col_name, solved_date="Løst_dato", solved_before=0):
    df.loc[:, solved_date] = pd.to_datetime(df[solved_date], errors="coerce")
    df.drop(
        df[
            df[solved_date] + pd.DateOffset(solved_before) <= df[df_date_col_name]
        ].index,
        inplace=True,
    )
    return df


def convert_numeric_col(df, num_col, var_name, conv_factor, decimals):
    # df = df[pd.to_numeric(df[num_col], errors='coerce').notnull()]
    try:
        df.loc[:, var_name] = (df[num_col].astype(float) * conv_factor).round(decimals)
    except:
        df.loc[:, var_name] = df[num_col]
    return df


def filt_delta_var(df, min_val=None, max_val=None, delta_var="days"):
    delta_var = "delta_" + delta_var
    if max_val == None:
        df = df[df[delta_var] >= min_val].sort_values(by=["CPR_hash", delta_var])
    elif min_val == None:
        df = df[df[delta_var] <= max_val].sort_values(by=["CPR_hash", delta_var])
    elif min_val != None and max_val != None:
        df = df[df[delta_var].between(min_val, max_val, inclusive="both")].sort_values(
            by=["CPR_hash", delta_var]
        )
    else:
        print("Specify at least either min or max value")
    return df


def get_latest_var(df, time_var, delta_var="days"):
    delta_var = "delta_" + delta_var
    df.loc[:, "delta_abs"] = df[delta_var].abs()
    df.sort_values(["CPR_hash", "delta_abs"], inplace=True)
    df.drop_duplicates(["CPR_hash", time_var], inplace=True)
    df.drop(columns=["delta_abs"], inplace=True)
    return df


def add_to_base_df(base_df, base_df_date_col_name, input_df, col_name, to_binary=False):
    """
    Adds a variable from input_df to base_df, either as (pseudo) binary or value as is.

    Parameters
    ----------
    col_name: str
        name for the new column to be added
    """

    # in case of using this function with input_df created/filtered in a loop
    if input_df.empty:
        base_df.loc[:, col_name] = 0
        return base_df

    else:
        if to_binary:
            input_df.loc[:, col_name] = 1
            output_df = base_df.merge(
                input_df[["CPR_hash", base_df_date_col_name, col_name]],
                on=["CPR_hash", base_df_date_col_name],
                how="left",
            ).drop_duplicates()
            output_df[col_name].fillna(0, inplace=True)
            output_df.loc[:, col_name] = output_df[col_name].astype(int)

        else:
            # base_df.drop(columns = col_name, inplace = True, errors='ignore')
            output_df = base_df.merge(
                input_df[["CPR_hash", base_df_date_col_name, col_name]],
                how="left",
                on=["CPR_hash", base_df_date_col_name],
            )

        return output_df


## High level transformation
def add_categorical_variable(
    base_df,
    df,
    filt_date,
    var_name,
    base_df_date_col_name,
    df_date_col_name=None,
    filt_column=None,
    filt_val=None,
    solved_date=None,
    solved_before=0,
    min_val=0,
    max_val=None,
    delta_var="days",
):
    """
    Adding a categorical variable to base_df from df in the following steps:

    - Filtering df columns by filt_var (tupple or string)
    - Getting delta_time columns (mode can either be days, hours or minutes)
    - Filter df based on delta_time rules (e.g only -10 to +10 hours from baseline datetime)
    - Filter df to get the row closest to baseline datetime
    - Adds the variable to base_df and returns

    Parameters
    ----------
    base_df : pandas dataframe
        target df where new variable should be added

    df: pandas dataframe
        df containing the new variable

    filt_date: str
        colunmn name of df datetime used for calculating delta days

    var_name: str
        name for new continous column in base_df

    filt_column: str
        column name of df to be filtered (e.g "fruit_type")

    filt_val: tuple | string
        inclusive values to filt_column (e.g ("apple", "banana", "mango") )
    """

    # filtering df. Try tuple else string
    try:
        # backup_df = df.copy(deep=True)
        df = df[df[filt_column].str.startswith(filt_val, na=False)]
    except:
        df = df[df[filt_column] == filt_val]

    if df.empty:
        base_df.loc[:, var_name] = 0
        return base_df

    else:
        df[filt_date] = pd.to_datetime(df[filt_date])
        base_df[base_df_date_col_name] = pd.to_datetime(base_df[base_df_date_col_name])

        df = get_delta_time(
            base_df,
            df,
            base_df_date=base_df_date_col_name,
            input_df_date=filt_date,
            mode=delta_var,
        )

        if solved_date != None:
            df = remove_solved(df, base_df_date_col_name, solved_date, solved_before)

        df = filt_delta_var(df, min_val, max_val, delta_var)

        df = get_latest_var(df, base_df_date_col_name, delta_var)

        df = add_to_base_df(
            base_df, base_df_date_col_name, df, var_name, to_binary=True
        )

        return df


def add_continous_variable(
    base_df,
    df,
    filt_column,
    filt_val,
    filt_date,
    num_col,
    var_name,
    base_df_date_col_name,
    conv_factor=None,
    decimals=2,
    min_val=0,
    max_val=None,
    delta_var="days",
):
    """
    Adding a continous variable to base_df from df in the following steps:

    - Filtering df columns by filt_var (tupple or string)
    - Getting delta_time columns (mode can either be days, hours or minutes)
    - Filter df based on delta_time rules (e.g only -10 to +10 hours from baseline datetime)
    - Filter df to get the row closest to baseline datetime
    - Convert the numerical value and round decimals if needed
    - Adds the variable to base_df and returns

    Parameters
    ----------
    base_df : pandas dataframe
        target df where new variable should be added

    df: pandas dataframe
        df containing the new variable

    filt_column: str
        column name of df to be filtered (e.g "fruit_type")

    filt_var: tuple | string
        inclusive values to filt_column (e.g ("apple", "banana", "mango") )

    filt_date: str
        colunmn name of df datetime used for calculating delta days

    num_col: str
        column name of the numerical/continous variable

    var_name: str
        name for new continous column in base_df
    """
    # filtering df. Try tuple else string
    try:
        df = df[df[filt_column].isin(filt_val)]
    except:
        df = df[df[filt_column] == filt_val]

    df[filt_date] = pd.to_datetime(df[filt_date])
    base_df[base_df_date_col_name] = pd.to_datetime(base_df[base_df_date_col_name])
    # collecting delta_time column
    df = get_delta_time(
        base_df,
        df,
        base_df_date=base_df_date_col_name,
        input_df_date=filt_date,
        mode=delta_var,
    )

    # filtering on delta_var
    df = filt_delta_var(df, min_val, max_val, delta_var)

    # latest, min, max, mean?
    df = get_latest_var(df, base_df_date_col_name, delta_var=delta_var)

    df = convert_numeric_col(df, num_col, var_name, conv_factor, decimals)
    df = add_to_base_df(base_df, base_df_date_col_name, df, var_name, to_binary=False)

    return df
