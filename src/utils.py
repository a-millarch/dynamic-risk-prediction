import os
import pandas as pd

pd.options.mode.chained_assignment = None

import hashlib
import yaml


def get_cfg(cfg_path="conf/defaults.yaml"):
    with open(cfg_path) as file:
        return yaml.safe_load(file)


def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_columns_with_word(dataframe, word):
    matching_columns = [col for col in dataframe.columns if word in col]
    return matching_columns


def ensure_datetime(df, column_name):
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        # df = df[df[column_name].notnull()]
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    return df


def is_file_present(file_path: str):
    return os.path.isfile(file_path)


def are_files_present(directory, filenames, extension):
    return all(
        os.path.isfile(os.path.join(directory, f"{filename}{extension}"))
        for filename in filenames
    )


def clear_mem():
    import torch
    import gc

    # Delete: any unused Python objects
    gc.collect()

    # Clear the PyTorch CUDA cache
    torch.cuda.empty_cache()

    # Total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Allocated memory
    allocated_memory = torch.cuda.memory_allocated(0)

    # Cached memory
    cached_memory = torch.cuda.memory_reserved(0)

    # Free memory
    free_memory = total_memory - allocated_memory - cached_memory

    print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Cached GPU memory: {cached_memory / 1e9:.2f} GB")
    print(f"Free GPU memory: {free_memory / 1e9:.2f} GB")


# -----------------------------------------------------------------------------------------------------------
# Common tools
def inches_to_cm(inches):
    return inches * 2.54


def feet_to_cm(feet):
    return feet * 30.48


def pounds_to_kg(pounds):
    return pounds * 0.45359237


def ounces_to_kg(ounces):
    return ounces * 0.0283495231


def dict_to_list(input_dict):
    result = []
    for i, v in input_dict.items():
        for item in v:
            result.append(item)
    return result


def list_from_col(df, col_name, target_col="col_cat", return_col="TQIP_name"):
    return df[df[target_col] == col_name][return_col].tolist()
