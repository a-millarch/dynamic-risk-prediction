import pandas as pd
import numpy as np
import logging 

from src.data.comorbidity import add_from_diagnoses
from src.features.static_features import add_to_base
from src.features.static_features import add_iss, add_elixhauser
from src.utils import find_columns_with_word
from src.data.utils import stratified_split_dataframe, get_base_df
from src.common.log_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class TabDS:
    def __init__(self, cfg, default_mode=True, base=None, exclude="study"):
        self.cfg = cfg
        if base is None:
            self.base = get_base_df()
        else:
            self.base = base
        self.cols = {}
        self.concepts = {}

        if default_mode:
            self.base = add_to_base(self.base)
            if cfg["prehospital"]:
                # YO this is bad but how we do for now
                base_len = len(self.base)
                logger.info("Adding PPJ to base")
                ppj_base = pd.read_pickle("data/interim/ppj_base_df.pkl")
                keep_cols = [
                    "PID",
                    "prehospital_start",
                    "prehospital_end",
                    "A",
                    "B",
                    "C",
                    "D",
                ]
                self.base = self.base.merge(ppj_base[keep_cols], on="PID", how="left")
                assert base_len - len(self.base) == 0

            logger.debug(self.base.columns)
            logger.info("Loaded basefile and added static info")
            if exclude == "study":
                self.exclude(study=True)
            elif exclude == "minimal":
                self.exclude(study=False)

            self.add_scores()
            self.add_vitals()

    def exclude(self, study=True):
        df = self.base
        original_len = len(df)

        # Include only if relevant free-text notes
        meta = pd.read_csv("data/interim/base_meta.csv", index_col=0)
        inclusion_pids = meta[meta.TRAUMATEXT == True].PID.unique()
        df = df[df.PID.isin(inclusion_pids)]

        if study == True:

            df = df[
                (df.DURATION <= self.cfg["dataset"]["max_duration_days"])
                & (
                    df.AGE.between(
                        cfg["dataset"]["age_min"],
                        cfg["dataset"]["age_max"],
                        inclusive="both",
                    )
                )
                & (df.start.dt.year >= cfg["dataset"]["start_year"])
                & (df.start.dt.year <= cfg["dataset"]["end_year"])
            ].reset_index(drop=True)
        if self.cfg["prehospital_only"]:
            logger.info(f"> prehospital only going from {len(df)}")
            df = df[df.prehospital_start.notnull()]
            logger.info(f">> to {len(df)}")

        logger.info(
            f"Reduced base df from {original_len}"
            f" to {len(df)} due to inclusion criteria"
        )
        self.base = df

    def add_scores(self):
        """Add Elixhauser and ISS to base"""

        # ELIXHAUSER
        while True:
            try:
                elix = pd.read_csv(
                    "data/interim/ISS_ELIX/computed_elix_df.csv", low_memory=False
                )
                logger.info("Elixhauser df dataframe found, continuing")
                baselen = len(self.base)
                # merge
                self.base = self.base.merge(
                    elix[["PID", "elixscore"]], how="left", on="PID"
                )
                assert baselen - len(self.base) == 0
                logger.info("Merged Elix onto base")
            # TODO: merge onto base
            except FileNotFoundError:
                logger.info("No Elixhauser computed, creating.")
                add_elixhauser(self.base)
                continue
            break

        # ISS
        while True:
            try:
                iss = pd.read_csv(
                    "data/interim/ISS_ELIX/computed_iss_df.csv", low_memory=False
                )
                logger.info("ISS df dataframe found, continuing")
                baselen = len(self.base)
                # Now merge
                self.base = self.base.merge(
                    iss[["PID", "riss", "maxais"]], how="left", on="PID"
                )
                assert baselen - len(self.base) == 0
                self.base[["riss", "maxais"]] = self.base[["riss", "maxais"]].replace(
                    "None", np.nan
                )
                logger.info("Merged ISS onto base")
            except:
                logger.info("No ISS results, creating.")
                add_iss(self.base)
                continue
            break

    def add_vitals(self, drop_cols: list = cfg["drop_features"]["VitaleVaerdier"]):
        self.vitals = {}
        for agg_func in cfg["agg_func"]["VitaleVaerdier"]:
            logger.info(f"Loading vitals.agg_func: {agg_func}")
            vit = pd.read_csv(
                f"data/interim/mapped/VitaleVaerdier_{agg_func}.csv", index_col=0
            )

            vit = vit[
                ~vit.FEATURE.isin(
                    drop_cols
                    + [
                        np.nan,
                    ]
                )
            ]
            vit["VALUE"] = pd.to_numeric(vit["VALUE"], errors="coerce")

            # Group by PID and unstack
            grouped = vit.groupby(["PID", "FEATURE"])["VALUE"].agg(
                ["first", "mean", "min", "max", "std", "count"]
            )
            unstacked = grouped.unstack(level="FEATURE")

            # Flatten and rename the column names
            unstacked.columns = [
                f"{stat.upper()}_{feature}" for feature, stat in unstacked.columns
            ]

            # Sort columns, reset index and merge
            sorted_columns = sorted(unstacked.columns)
            unstacked = unstacked[sorted_columns].reset_index()

            self.vitals[agg_func] = unstacked

    def add_concept(self, concept):
        cfg = self.cfg
        drop_cols = cfg["drop_features"][concept]

        self.concepts[concept] = {}

        for agg_func in cfg["agg_func"][concept]:
            logger.info(f"Loading {concept}.agg_func: {agg_func}")
            df = pd.read_csv(
                f"data/interim/mapped/{concept}_{agg_func}.csv", index_col=0
            )
            df = df[
                ~df.FEATURE.isin(
                    drop_cols
                    + [
                        np.nan,
                    ]
                )
            ]
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

            # Group by PID and unstack
            grouped = df.groupby(["PID", "FEATURE"])["VALUE"].agg(
                ["first", "mean", "min", "max", "std", "count"]
            )
            unstacked = grouped.unstack(level="FEATURE")

            # Flatten and rename the column names
            unstacked.columns = [
                f"{stat.upper()}_{feature}" for feature, stat in unstacked.columns
            ]

            # Sort columns, reset index and merge
            sorted_columns = sorted(unstacked.columns)
            unstacked = unstacked[sorted_columns].reset_index()

            self.concepts[concept][agg_func] = unstacked

    def add_diagnoses(self, merge=False):
        diag = pd.read_pickle("data/interim/Diagnoser.pkl")
        self.hc_cc = add_from_diagnoses(
            diag, self.base[["CPR_hash", "PID", "ServiceDate"]].copy(deep=True)
        )

        if merge:
            self.base = self.base.merge(
                self.hc_cc, on=["CPR_hash", "PID", "ServiceDate"], how="left"
            )
        # add column name lists
        for colprefix in ["CC", "HC"]:
            self.cols[colprefix] = find_columns_with_word(self.hc_cc, colprefix)

    def add_labs(self):
        pass

    def get_eda_df(self, target="deceased_90d", TB_only=False, vital_criteria=True):

        poplist = [
            "CPR_hash",
            "start",
            "end",
            "first_afsnit",
            "first_RH",
            "time_to_RH",
            "type_visitation",
            "ServiceDate",
            "DOB",
            "DOD",
            "trajectory",
            "overlap",
        ]

        df = self.base.copy(deep=True)
        df = df.rename(
            columns={
                "elixscore": "ASMT_ELIX",
                "maxais": "ASMT_MAXAIS",
                "riss": "ASMT_ISS",
                "TB": "LVL1TC",
            }
        )

        for agg_func in cfg["agg_func"]["VitaleVaerdier"]:
            vit = self.vitals[agg_func]

            # new_columns = [vit.columns[0],] + [col + '_'+ agg_func for col in vit.columns[1:]]
            # vit.columns = new_columns

            # Add vitals
            if agg_func == "std":

                new_columns = [
                    vit.columns[0],
                ] + [col + "_" + agg_func for col in vit.columns[1:]]
                vit.columns = new_columns
                a = find_columns_with_word(vit, "_max_")
                keep_cols = a
            else:
                a = find_columns_with_word(vit, f"{agg_func}")
                # c = find_columns_with_word(vit, '_count_')
                keep_cols = a  # +c
            # keep_cols = find_columns_with_word(self.vital_df, '_first')

            df = df.merge(
                vit[
                    sorted(
                        [
                            "PID",
                        ]
                        + keep_cols
                    )
                ],
                on="PID",
                how="left",
            )
            # remove cols not needed and update lists
        # if only trauma bay
        if TB_only:
            df = df[df.LVL1TC == 1].reset_index(drop=True)
            # df=df.drop(columns='LVL1TC')
            # cat_cols.remove('LVL1TC')

        if vital_criteria:
            # use count of mean for inclusion
            vit_filter_df = self.vitals["mean"]
            # minimum bins with values based on SBP
            vit_filter_pids = vit_filter_df[
                vit_filter_df["SBP_count"] >= self.cfg["dataset"]["min_bin_seq_len"]
            ]["PID"].unique()
            df_len = len(df)
            df = df[df.PID.isin(vit_filter_pids)].reset_index(drop=True)
            logger.info(f"dropped another {df_len - len(df)} pids")

            # first_key = next(iter(self.vitals))
            # df = df[df.PID.isin(self.vitals[first_key].PID.unique())].reset_index(drop=True)

        # Mark holdout
        if self.cfg["holdout_type"] == "temporal":
            logger.info("temporal holdout split")
            holdout_pids = df.sort_values("ServiceDate", ascending=False).head(
                int(len(df) * self.cfg["holdout_fraction"])
            )["PID"]
            df.loc[df["PID"].isin(holdout_pids), "HOLDOUT"] = True

        elif self.cfg["holdout_type"] == "random":
            logger.info("random(seeded) stratified holdout split")
            df = stratified_split_dataframe(
                df,
                self.cfg["target"],
                test_size=self.cfg["holdout_fraction"],
                random_state=self.cfg["seed"],
            )

        else:
            logger.error("No holdout split was done!")
        self.eda_df_raw = df.copy(deep=True)
        # Reduce cols
        drop = ["ASMT_MAXAIS"]  #'deceased_30d',
        df = df.drop(columns=drop + poplist)

        exclude = ["PID", "HOLDOUT", "deceased_30d"]

        cat_cols = cfg["dataset"]["cat_cols"]
        # cat_cols = ['SEX', 'LVL1TC']
        ppj_cat_cols = cfg["dataset"]["ppj_cat_cols"]
        # ppj_cat_cols = ['A: Luftveje', 'B: Respiration', 'C: Cirkulation', 'D: Bevidsthedsniveau']
        cat_cols = cat_cols + ppj_cat_cols
        num_cols = [
            i
            for i in df.columns
            if i not in cat_cols and i != target and i not in exclude
        ]

        # TODO: fix class boilerplate
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target = target
        self.eda_df = df

        return df.copy(deep=True)

    def get_analysis_df(self, target="deceased_90d", vital_criteria=True):

        if not hasattr(self, "eda_df"):
            self.get_eda_df(
                target=target,
                TB_only=self.cfg["TB_only"],
                vital_criteria=vital_criteria,
            )

        df = self.eda_df
        # remove counts from model, used for EDA
        # counts = find_columns_with_word(df, 'count')
        num_drops = [
            "DURATION",
        ]

        # df = df.drop(columns=counts+num_drops)
        drop_cols = num_drops  # +counts
        df = df.drop(columns=drop_cols)
        # hacky solution to running this function several times
        try:
            for i in drop_cols:
                self.num_cols.remove(i)
        except:
            pass
        self.analysis_df = df
        # export for GPU kernel, separate for trainval and holdout
        val_df = self.eda_df_raw
        val_df = val_df[val_df.HOLDOUT == True]

        holdout = df[df.PID.isin(val_df.PID.unique())].reset_index(drop=True)
        df = df[~df.PID.isin(val_df.PID.unique())].reset_index(drop=True)

        df.to_csv("data/processed/trainval_tab_simple.csv")
        holdout.to_csv("data/processed/holdout_tab_simple.csv")

        logger.info(
            f"Train/val df shape: {df.shape}\n with {df[target].sum()}/{((df[target].sum()/len(df))*100).round(2)}% outcomes"
        )
        logger.info(
            f"Holdout df shape: {holdout.shape}\n with {holdout[target].sum()}/{((holdout[target].sum()/len(holdout))*100).round(2)}% outcomes"
        )
        self.trainval_df = df
        self.holdout_df = holdout
