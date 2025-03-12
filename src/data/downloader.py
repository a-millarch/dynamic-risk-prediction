import os
from azureml.core import Workspace, Datastore, Dataset
from src.utils import is_file_present


def download_to_local(FILES: list, LOCAL_DIR="data/dl"):
    #### Step 1: Download
    WS = Workspace.from_config()

    ## Get datastores
    datastore_sp = Datastore.get(WS, "sp_data")

    total_failed = 0
    os.makedirs(LOCAL_DIR, exist_ok=True)
    f_status = dict()
    for fn in FILES:
        if is_file_present(f"data/dl/CPMI_{fn}.parquet"):
            pass
        else:
            try:
                print("> ", fn)
                ds = Dataset.File.from_files((datastore_sp, "CPMI_" + fn + ".parquet"))
                print(">> Downloading...")
                ds.download(LOCAL_DIR)
                print(">> Done!")
                f_status[fn] = True
            except:
                f_status[fn] = False
                total_failed += 1
                print(">> Failed!!!")
                print(f"Could not load {fn}!", False)


if __name__ == "__main__":
    download_to_local()
