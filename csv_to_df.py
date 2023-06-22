import pandas as pd
from pathlib import Path
from sys import platform


def csv_to_df(
    file_folder,
    file_name,
    channel_names=["channel 1", "channel 2", "channel 3", "wave_gen"],
):
    """Returns a DataFrame from a .csv file.
    Set channel_names to None to return a
    DataFrame with the default column names.
    """
    if platform == "linux":
        ROOT_FOLDER = "/mnt/c/Users/nikla/OneDrive - NTNU/NTNU/ProsjektOppgave"
        file_path = f"{ROOT_FOLDER}/{file_folder}/{file_name}.csv"
    else:
        ROOT_FOLDER = f"{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave"
        file_path = f"{ROOT_FOLDER}\\{file_folder}\\{file_name}.csv"

    df = pd.read_csv(filepath_or_buffer=file_path, names=channel_names)
    print(
        "\n"
        + f'Creating DataFrame from file folder "{file_folder}" and file name "{file_name}'
        + "\n"
    )

    return df


def csv_to_df_thesis(
    file_folder,
    file_name,
    channel_names=["channel 1", "channel 2", "channel 3", "wave_gen"],
    scope=False,
    milli=True,
):
    """Returns a DataFrame from a .csv file.
    Set channel_names to None to return a
    DataFrame with the default column names.
    """
    if platform == "linux":
        ROOT_FOLDER = "/mnt/c/Users/nikla/OneDrive - NTNU/NTNU/Masteroppgave\spring2023"
        file_path = f"{ROOT_FOLDER}/{file_folder}/{file_name}.csv"
    else:
        ROOT_FOLDER = f"{Path.home()}\\OneDrive - NTNU\\NTNU\\Masteroppgave\spring2023"
        file_path = f"{ROOT_FOLDER}\\{file_folder}\\{file_name}.csv"
    if scope:
        df = pd.read_csv(filepath_or_buffer=file_path, skiprows=1)
    else:
        df = pd.read_csv(filepath_or_buffer=file_path, names=channel_names)
        conversion_factor = 100
        df = df.applymap(lambda x: x / conversion_factor)
        if milli:
            df = df.applymap(lambda x: x * 1000)
        # correct for amplifier gain on all channels to get m/s^2, by dividing by 100

    print(
        "\n"
        + f'Creating DataFrame from file folder "{file_folder}" and file name "{file_name}'
        + "\n"
    )

    return df


if __name__ == "__main__":
    csv_to_df("base_data", "df_average_noise")
