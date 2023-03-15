"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import os
import numpy as np
import pandas as pd


def csv_to_df(
    file_folder: str,
    file_name: str,
    channel_names: np.ndarray = np.array(
        ["Sensor 1", "Sensor 2", "Sensor 3", "Actuator"]
    ),
):
    """Returns a DataFrame from a .csv file.
    Set channel_names to None to return a
    DataFrame with the default column names.
    """
    file_path = os.path.join("Measurements", file_folder, f"{file_name}.csv")
    dataframe = pd.read_csv(filepath_or_buffer=file_path, names=channel_names)

    """Move 'Actuator' column to the front of the DataFrame"""
    columns = dataframe.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    dataframe = dataframe[columns]

    return dataframe


if __name__ == "__main__":
    csv_to_df("base_data", "df_average_noise")
