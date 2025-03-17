import pandas as pd
from src.data.utils import cfg, get_base_df

from src.dataset.tabular import TabDS
import logging

from src.data.utils import get_base_df, create_enumerated_id
from src.common.log_config import setup_logging
from src.data.collectors import collect_procedures, population_filter_parquet
from src.utils import ensure_datetime

setup_logging()
logger = logging.getLogger(__name__)


############ helpers
def find_forløb_original(
    overview: pd.DataFrame, target_df: pd.DataFrame, date_colname: str
) -> pd.DataFrame:

    CPR_list = set(target_df.CPR_hash)
    for CPR_hash in CPR_list:

        tmp_df = overview[overview["CPR_hash"] == CPR_hash]
        tmp_target_df = target_df[target_df["CPR_hash"] == CPR_hash]

        for f in tmp_df["trajectory"]:
            tmp_f_df = tmp_df[tmp_df.trajectory == f]

            for d in tmp_target_df[date_colname]:
                if (
                    d + pd.DateOffset(days=1) > pd.to_datetime(tmp_f_df.start.values)
                ) & (d < pd.to_datetime(tmp_f_df.end.values)):
                    target_df.loc[
                        (target_df["CPR_hash"] == CPR_hash)
                        & (target_df[date_colname] == d),
                        "trajectory",
                    ] = f
    return target_df


def find_forløb(
    base: pd.DataFrame, df: pd.DataFrame, dt_name: str, offset=1
) -> pd.DataFrame:
    # save colnames for return
    colnames = df.columns.to_list()
    # ensure datetime format for input df
    df = ensure_datetime(df, dt_name)
    # merge and filter
    merged_df = base.merge(df, on="CPR_hash", how="left")

    filtered_df = merged_df[
        (merged_df[dt_name] >= merged_df["start"] - pd.DateOffset(days=offset))
        & (merged_df[dt_name] <= merged_df["end"] + pd.DateOffset(days=offset))
    ]
    filtered_df = filtered_df.drop_duplicates().reset_index(drop=True)

    return filtered_df[colnames + ["trajectory"]]


def check_overlaps(group):
    """Checks for overlapping in start - end of trajectories for same patient in groupby object

    usage example:
    df['overlap'] = df.groupby('CPR_hash').apply(check_overlaps).explode().reset_index(drop=True)

    """
    overlaps = []
    for i in range(len(group) - 1):
        # Check if the current end_time overlaps with the next start_time
        if group.iloc[i]["end"] > group.iloc[i + 1]["start"]:
            overlaps.append(True)
        else:
            overlaps.append(False)
    # Append False for the last entry as it has no next entry to compare
    overlaps.append(False)
    return overlaps


def collapse_admissions(df):
    """
    Collapse hospital admissions based on time intervals.

    Parameters:
    df (pd.DataFrame): DataFrame containing patient admission data with columns:
                       ID, trajectory, start, end.

    Returns:
    pd.DataFrame: A new DataFrame with collapsed admissions.
    """

    # Ensure datetime format for start and end columns
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Calculate duration and delta_nplus_trajectory
    df["duration"] = df["end"] - df["start"]
    df["next_trajectory_start"] = df["start"].shift(-1)
    df["delta_nplus_trajectory"] = df["next_trajectory_start"] - df["end"]

    # Initialize a list to hold collapsed rows
    collapsed_rows = []

    # Initialize a variable to track the current group
    current_group = []

    for index, row in df.iterrows():
        if not current_group:
            current_group.append(row)
        else:
            # Check if the delta is less than 1 hour
            last_row = current_group[-1]
            if (row["start"] - last_row["end"]) < pd.Timedelta(hours=1) and row[
                "CPR_hash"
            ] == last_row["CPR_hash"]:
                current_group.append(row)
            else:
                # Collapse the current group and reset
                collapsed_start = current_group[0]["start"]
                collapsed_end = current_group[-1]["end"]
                total_duration = collapsed_end - collapsed_start

                # Create a new row for the collapsed group
                collapsed_rows.append(
                    {
                        "CPR_hash": last_row["CPR_hash"],
                        "trajectory": ",".join(
                            map(str, [row["trajectory"] for row in current_group])
                        ),
                        "start": collapsed_start,
                        "end": collapsed_end,
                        "duration": total_duration,
                    }
                )
                # Start a new group with the current row
                current_group = [row]

    # Handle any remaining groups after the loop ends
    if current_group:
        collapsed_start = current_group[0]["start"]
        collapsed_end = current_group[-1]["end"]
        total_duration = collapsed_end - collapsed_start

        collapsed_rows.append(
            {
                "CPR_hash": current_group[0]["CPR_hash"],
                "trajectory": ",".join(
                    map(str, [row["trajectory"] for row in current_group])
                ),
                "start": collapsed_start,
                "end": collapsed_end,
                "duration": total_duration,
            }
        )

    # Create final DataFrame from collapsed data
    final_df = pd.DataFrame(collapsed_rows)

    return final_df


#################################
def create_base_df(cfg):
    """Create the base dataframe from scratch"""

    logger.info("Starting the creation of the base dataframe.")

    # Load or collect procedure inclusion file (BWST1F)
    while True:
        try:
            logger.info("Loading procedure inclusion file.")
            population = pd.read_csv("data/raw/Procedurer_population.csv", index_col=0)
            logger.info("Procedure inclusion file loaded successfully.")
        except FileNotFoundError:
            logger.warning("Procedure file not found, collecting and retrying.")
            collect_procedures(cfg)
            continue
        break

    population["ServiceDate"] = pd.to_datetime(
        population["ServiceDate"], format="mixed"
    )

    # Load or collect ADT events for population
    while True:
        try:
            logger.info("Loading ADT events file.")
            df_ad = pd.read_csv(
                "data/raw/ADTHaendelser.csv", dtype={"CPR_hash": str}, index_col=0
            )
            logger.info("ADT events file loaded successfully.")
        except FileNotFoundError:
            logger.warning("ADT file not found, collecting and retrying.")
            population_filter_parquet("ADTHaendelser", cfg=cfg, base=population)
            continue
        break

    # Convert to datetime
    logger.info("Converting datetime columns.")
    df_ad[["Flyt_ind", "Flyt_ud"]] = df_ad[["Flyt_ind", "Flyt_ud"]].apply(
        lambda x: pd.to_datetime(x, format="mixed", errors="coerce")
    )

    # Adjust 'Flyt Ind' event times by 1 second
    logger.info("Adjusting 'Flyt Ind' event times.")
    df_ad.loc[df_ad.ADT_haendelse == "Flyt Ind", "Flyt_ind"] += pd.Timedelta(seconds=1)

    # Sort and compute trajectories
    logger.info("Sorting and computing trajectories.")
    df_ad = df_ad.sort_values(["CPR_hash", "Flyt_ind"])
    df_ad["trajectory"] = (
        df_ad.loc[df_ad["ADT_haendelse"] == "Indlæggelse"]
        .groupby("CPR_hash")
        .cumcount()
        + 1
    )
    df_ad["trajectory"] = df_ad["trajectory"].ffill()

    # Compute start, end, and duration of trajectories
    logger.info("Computing start, end, and duration of trajectories.")
    grouped = df_ad.groupby(["CPR_hash", "trajectory"])
    of = grouped.Flyt_ind.min().to_frame(name="start").reset_index()
    of["end"] = grouped.Flyt_ud.max().values
    of["duration"] = of["end"] - of["start"]
    of["next_trajectory_start"] = of.groupby("CPR_hash")["start"].shift(-1)
    of["delta_nplus_trajectory"] = of["next_trajectory_start"] - of["end"]

    # grace period 1 hour as is between trajectories
    logger.info("Collapsing admissions with less than an hour between each other")
    of = collapse_admissions(of)

    # Add trajectory, merge with population data
    date_colname = "ServiceDate"
    # TODO: set of = and not fdf? Not sure why this works.. Rewrite function completely, it is slow
    fdf = find_forløb(of, population, date_colname)
    logger.info("Merging with population data.")
    df = pd.merge(
        fdf[["CPR_hash", "trajectory", date_colname]],
        of,
        on=["CPR_hash", "trajectory"],
        how="left",
    )

    logger.info("Saving tmp dataframe to a pickle file.")
    df.to_pickle("data/interim/tmp_base_df.pkl")
    # Sort df_adt for further processing
    df_adt = df_ad.sort_values(by=["CPR_hash", "Flyt_ind"])

    ### FIND FIRST AFSNIT
    logger.info("Finding first department contact")
    df_tra2 = df[["CPR_hash", "ServiceDate", "start", "end"]].copy()  # 'trajectory',
    merged_df = pd.merge(df_adt, df_tra2, on="CPR_hash")
    # Filter events within the start and end time range
    filtered_events = merged_df[
        (merged_df["Flyt_ind"] >= merged_df["start"])
        & (merged_df["Flyt_ind"] <= merged_df["end"])
    ]
    # Find the first value for each patient
    first_values = (
        filtered_events.groupby(["CPR_hash", "ServiceDate", "start"])
        .first()
        .reset_index()
    )

    ### FIRST RH
    logger.info("Finding first RH-TC contact")
    filtered_RH = filtered_events[
        filtered_events["Afsnit"].str.contains("RH ", case=False, na=False)
    ]
    # Find the first value for each patient
    first_RH = (
        filtered_RH.groupby(["CPR_hash", "ServiceDate", "start"]).first().reset_index()
    )
    # Select relevant columns
    first_RH = first_RH[["CPR_hash", "Flyt_ind", "ServiceDate", "start"]].rename(
        columns={"Flyt_ind": "first_RH"}
    )
    result = pd.merge(
        first_values, first_RH, how="left", on=["CPR_hash", "ServiceDate", "start"]
    ).rename(columns={"Afsnit": "first_afsnit"})
    # Add time to RH and visitation type
    result["time_to_RH"] = result.first_RH - result.start
    result["type_visitation"] = None

    ### Set visitation type
    logger.info("Adding visitation type")
    result.loc[
        result["first_afsnit"].str.contains("RH TRAUMECENTER"), "type_visitation"
    ] = "primær"

    result.loc[
        (~result["first_afsnit"].str.contains("RH TRAUMECENTER"))
        & (result["first_RH"].notnull()),
        "type_visitation",
    ] = "sekundær"
    # (result['time_to_RH'] > pd.Timedelta(days=0)), 'type_visitation'] = 'sekundær'

    result["type_visitation"] = result["type_visitation"].fillna("primær ingen RH")

    # Mark overlaps
    result["overlap"] = (
        result.groupby("CPR_hash")
        .apply(check_overlaps)
        .explode()
        .reset_index(drop=True)
    )

    # Load and merge patient information
    logger.info("Loading and merging patient information.")
    population_filter_parquet("PatientInfo", base=population)
    pi = pd.read_csv("data/raw/PatientInfo.csv", index_col=0)
    pi = pi.rename(
        columns={"Fødselsdato": "DOB", "Dødsdato": "DOD", "Køn": "SEX"}
    ).replace(
        {"Mand": "Male", "Kvinde": "Female", "Ukendt": "Male"}
    )  # ugh I know
    result = result.merge(
        pi[["CPR_hash", "DOB", "DOD", "SEX"]], on="CPR_hash", how="left"
    )
    logger.info("Patient info data loaded and merged successfully.")

    # Add PID
    logger.info("Adding PID")
    result = create_enumerated_id(result, "CPR_hash", "ServiceDate")
    # cleanup
    result = result[(result["start"].notnull()) & (result["end"].notnull())]
    ##### tmp
    logger.info("Saving the dataframe to a pickle file.")
    result.to_pickle("data/interim/base_df.pkl")
    ##### tmp
    result.drop_duplicates(subset="PID", inplace=True)

    result = result.drop(columns=["Flyt_ind", "Flyt_ud", "ADT_haendelse"])
    result = result.reset_index(drop=True)

    #### tmp, done manually to base:df 29.10.2024
    result = result.drop_duplicates(subset=["CPR_hash", "start", "end"])
    result = result.reset_index(drop=True)

    # Save to file
    logger.info("Saving the dataframe to a pickle file.")
    result.to_pickle("data/interim/base_df.pkl")
    logger.info("Base dataframe creation completed successfully.")


def create_bin_df(cfg):
    bin_list = []
    base = get_base_df()

    # Load bin intervals from cfg
    bin_intervals = cfg["bin_intervals"]

    for _, row in base.iterrows():
        start_time = row["start"]
        end_time = row["end"] + pd.Timedelta(minutes=10)
        pid = row["PID"]

        current_time = start_time
        bin_counter = 1

        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break

            # Determine the end time for this interval
            if interval == "end":
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)

            # Create bins for this interval
            bins = pd.date_range(
                start=current_time,
                end=min(interval_end, end_time),
                freq=freq,
                inclusive="left",
            )

            # Add bins to the list
            bin_list.extend(
                [
                    (pid, bin_start, bin_end, bin_counter + i, freq)
                    for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:]))
                ]
            )

            # Update the current time and bin counters
            current_time = bins[-1]
            bin_counter += len(bins) - 1

    # Create DataFrame from bin list
    bin_df = pd.DataFrame(
        bin_list, columns=["PID", "bin_start", "bin_end", "bin_counter", "bin_freq"]
    )

    # Save DataFrame to pickle file
    bin_df.to_pickle(cfg["bin_df_path"])


def load_ppj_mapping():
    
    from azureml.core import  Dataset

    # get mapping for merging data
    ppj_path =  'ppj.csv'
    ppj = Dataset.Tabular.from_delimited_files(path= ppj_path, separator=';').to_pandas_dataframe()
    ppj.drop(columns="Column1", inplace=True)
    ppj.drop_duplicates(inplace=True)
    return ppj

def load_ppj_data():
    # Temp function to concat raw PPJ data. Full PPJ pending upload to Azure
    ppj_data = []
    # note, 2025 file is of course just a tmp fix to missing people
    for year in range(2017,2027):
        print(year)
        ppj_data.append(pd.read_csv(f'data/dumps/ppj/{year}.csv', sep=';', index_col=0))

    ppj = pd.concat(ppj_data)

def create_bin_df_ppj(base, bin_intervals, start_colname, end_colname, save_path=None):
    bin_list = []
   
    for _, row in base.iterrows():
        start_time = row[start_colname]
        end_time = row[end_colname] + pd.Timedelta(minutes=10)
        pid = row['PID']
        
        current_time = start_time
        bin_counter = 1
        
        for interval, freq in bin_intervals.items():
            if current_time >= end_time:
                break
            
            # Determine the end time for this interval
            if interval == 'end':
                interval_end = end_time
            else:
                interval_end = start_time + pd.Timedelta(interval)
            
            # Create bins for this interval
            bins = pd.date_range(start=current_time, end=min(interval_end, end_time), freq=freq, inclusive='left')
            
            # Add bins to the list
            bin_list.extend([(pid, bin_start, bin_end, bin_counter + i, freq) 
                             for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:]))])
            
            # Update the current time and bin counters
            current_time = bins[-1]
            bin_counter += len(bins) - 1
            
    # Create DataFrame from bin list
    bin_df = pd.DataFrame(bin_list, columns=['PID', 'bin_start', 'bin_end', 'bin_counter', 'bin_freq'])
    
    # Save DataFrame to pickle file
    if save_path:
        bin_df.to_pickle(save_path)
    return bin_df

def ppj_proces():
    # Requires trIAIge project as ppj
    from ppj.data.datasets import ppjDataset,TabularDataset
    # load base and prepare data
    prePH_base = pd.read_pickle('data/interim/base_df.pkl')

    cfg["prehospital"] = False
    cfg["prehospital_only"] = False

    ds = TabDS(cfg, base= prePH_base)
    basecols = ['CPR_hash', 'ServiceDate', 'start', 'first_afsnit', 'trajectory', 'end',
           'first_RH', 'time_to_RH', 'type_visitation', 'overlap', 'DOB', 'DOD',
           'SEX', 'PID']
    base = ds.base[basecols]

    ppj_map = load_ppj_mapping()

    ppjpop = ppj_map[ppj_map.CPR_hash.isin(base.CPR_hash)]
    logger.info(f"Unique CPR's in ppj: {len(ppjpop['CPR_hash'].unique())}")
    # Merge mapping to base
    ph = base.merge(ppjpop, on='CPR_hash', how='outer')
    ph["CreationTime"] = ph["CreationTime_dt"].dt.floor('h')
    ph["CreationTime_dt"] = ph["CreationTime_dt"].dt.floor('d')
    ph["delta_hours_start"] = (pd.to_datetime(ph["CreationTime"])- pd.to_datetime(ph['start'])).dt.total_seconds()/(60*60)
    ph_pop = ph[(ph.CreationTime_dt<= ph.end) & (ph.delta_hours_start >= -48)].drop_duplicates()

    ppj = load_ppj_data()

    # Extract relevant variables
    ppj_ds = ppjDataset(mode='input', 
                        patient_info_file=ph_pop.copy(deep=True),    
                        ppj_file=ppj.drop_duplicates().copy(deep=True))

    # Extract GCS
    gcs = ppj_ds.subset_numericals["GCS"].copy(deep=True)
    try: 
        gcs.ManualTime.fillna(gcs.CreationTime, inplace=True)
        gcs["CreationTime"] = gcs ["ManualTime"]
        del gcs["ManualTime"]
    except:
        logger.warning(f"No column manualtime for GCS, proceed")

    gcs["FEATURE"] = "GCS"
    gcs = gcs.rename(columns = {"CreationTime":"TIMESTAMP", "Value":"VALUE"}).drop(columns=["JournalID", "EventCodeName"])
    gcs["TIMESTAMP"] = pd.to_datetime(gcs["TIMESTAMP"] ,format='mixed')
    gcs.to_pickle('data/interim/prehospital_GCS.pkl')

    #Vitals 
    ts_subset_names = ['M_NInv Sys Blodtryk', 'M_Puls','M_SpO2','M_NInv Dia Blodtryk']
    ts_subsets = []
    for s in ts_subset_names:
        ppj_ds.subset[s]["FEATURE"] = s

        ts_subsets.append(ppj_ds.subset[s])
    ts_subsets = pd.concat(ts_subsets)

    min_df = pd.DataFrame(ts_subsets.groupby("JournalID").CreationTime.min()).reset_index().rename(columns= {'CreationTime':'prehospital_start'})
    ph_pop = ph_pop.merge(min_df, on='JournalID', how='left')
    max_df = pd.DataFrame(ts_subsets.groupby("JournalID").CreationTime.max()).reset_index().rename(columns= {'CreationTime':'prehospital_end'})
    ph_pop = ph_pop.merge(max_df, on='JournalID', how='left')

    for dt in ["prehospital_start", "prehospital_end"]:
        ph_pop[dt] = pd.to_datetime(ph_pop[dt])

    ph_pop['prehospital_duration'] = ph_pop['prehospital_end'] -ph_pop['prehospital_start'] 
    ph_pop['prehospital_duration'].describe()

    ph_pop = ph_pop.drop(columns=["JournalID", "delta_hours_start", "CreationTime_dt","CreationTime",
                                 "prehospital_start", "prehospital_end", "prehospital_duration"]).drop_duplicates()

    # Now, same again but find start and end for each PID, not JID
    min_df = pd.DataFrame(ts_subsets.groupby("PID").CreationTime.min()).reset_index().rename(columns= {'CreationTime':'prehospital_start'})
    ph_pop = ph_pop.merge(min_df, on='PID', how='left')
    max_df = pd.DataFrame(ts_subsets.groupby("PID").CreationTime.max()).reset_index().rename(columns= {'CreationTime':'prehospital_end'})
    ph_pop = ph_pop.merge(max_df, on='PID', how='left')

    for dt in ["prehospital_start", "prehospital_end"]:
        ph_pop[dt] = pd.to_datetime(ph_pop[dt])

    # ABCD

    tab_subset_names =['A: Luftveje', 'B: Respiration', 'C: Cirkulation', 'D: Bevidsthedsniveau']
    tab_subsets = []
    for s in tab_subset_names:
        ppj_ds.subset[s]["FEATURE"] = s
        tab_subsets.append(ppj_ds.subset[s])
    tab_subsets = pd.concat(tab_subsets)

    # Formatting
    tab_ds = TabularDataset(base = ppj_ds, default_mode=False)

    for df in tab_ds.base.subset_categoricals.keys(): # type: ignore
        ss = tab_ds.base.subset_categoricals[df].copy(deep=True)
        print(df, ss.PID.duplicated().sum())

    tab_ds.merge_categoricals()
    rename_cols= {  "A: Luftveje_Value": "A",
                    "B: Respiration_Value" : "B", 
                    "C: Cirkulation_Value": "C", 
                    "D: Bevidsthedsniveau_Value":"D"}
    export_df = tab_ds.df.rename(columns=rename_cols)
    export_df = export_df.merge(ph_pop[["prehospital_start", "prehospital_end", "PID"]], on='PID', how='inner')

    assert export_df.PID.duplicated().sum() == 0
    assert len(export_df) == len(ph_pop)

    export_df.to_pickle('data/interim/ppj_base_df.pkl')


    # Add PH information to large base, use normal start if no PH for binning df
    base = prePH_base.merge(ph_pop[["PID","prehospital_start","prehospital_end"]], on='PID',how='left')
    base.loc[base.prehospital_start.isnull(), "prehospital_start"] = base.loc[base.prehospital_start.isnull()]["start"]

    bin_df = create_bin_df_ppj( base
                           ,cfg["bin_intervals"],'prehospital_start', 'end')

    bin_df.to_pickle('data/interim/bin_df.pkl')

    # now corrrect feature maps
    mapping = {"M_NInv Sys Blodtryk": "SBP", "M_NInv Dia Blodtryk":"DBP", "M_Puls":"HR", 'M_SpO2':'SPO2'}
    ts_subsets["FEATURE"] = ts_subsets["FEATURE"].replace(mapping)
    ts_subsets.rename(columns={"Value":"VALUE","CreationTime":"TIMESTAMP" }, inplace=True)
    ts_subsets = ts_subsets[(ts_subsets.PID.notnull()) & (ts_subsets.PID.isin(ph_pop.PID.unique()))]

    ts_subsets["TIMESTAMP"] = pd.to_datetime(ts_subsets["TIMESTAMP"] )
    ts_subsets[["TIMESTAMP","PID", "FEATURE", "VALUE"]].to_pickle('data/interim/prehospital_VitaleVaerdier.pkl')
