from __future__ import annotations

import json
import os


def read_json(fjson: str):
    """Read the json file.

    Args:
        fjson (str): file name of json to read.

    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as file:
        return json.load(file)


def write_json(d: dict, fjson: str):
    """Write the json file.

    Args:
        d (dict): dictionary to write
        fjson (str): file name of json to write.

    Returns:
        written dictionary
    """
    with open(fjson, "w") as file:
        json.dump(d, file)


def mkdir(path: str):
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path
