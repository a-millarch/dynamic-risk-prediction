import shutil
import os
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler

p = Path(__file__).parents[2]

current_date = datetime.now()
current_year = current_date.strftime("%Y")
current_month = current_date.strftime("%m")
current_day = current_date.strftime("%d")
current_time = current_date.strftime("%H%M")

default_log_file_path = "logging/app.log"
archive_folder = p.joinpath(
    f"logging/archive/{current_year}/{current_month}/{current_day}/"
)


def setup_logging():
    #   if not logging.getLogger().hasHandlers():

    formatter = logging.Formatter(
        "%(name)-22s - %(asctime)s - %(levelname)s - %(message)s"
    )
    handler = RotatingFileHandler(p.joinpath(default_log_file_path))
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    # change level here for now
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.root.handlers[0] = RichHandler(markup=True)

    # for some reason matplotlib is annoying in notebooks when level==debug. Disable
    logging.getLogger("matplotlib.font_manager").disabled = True


def clear_log():
    """Compatible with vscode Log Viewer extension"""
    log_file_path = p.joinpath(default_log_file_path)
    # Create the archive folder if it doesn't exist
    os.makedirs(archive_folder, exist_ok=True)
    # Get the current date in the format YYYYMMDD

    # Form the new file path in the archive folder with the current date suffix
    archive_file_path = p.joinpath(archive_folder, f"complete_{current_time}.txt")

    # Read the contents of the original log file
    with open(log_file_path, "r") as original_file:
        log_contents = original_file.read()

    # Write the contents to a new file in the archive folder
    with open(archive_file_path, "w") as archive_file:
        archive_file.write(log_contents)

    # Create a new empty log file in the original location
    with open(log_file_path, "w"):
        pass
