import pandas as pd
from pathlib import Path


def csv_to_df(file_folder, file_name, channel_names=['channel 1', 'channel 2', 'channel 3']):
    """Make a DataFrame from a .csv file"""

    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    file_path = ROOT_FOLDER + '\\' + file_folder + '\\' + file_name + '.csv'
    df = pd.read_csv(file_path, names=channel_names)
    print("\n" + "Using data file path:", file_path)

    return df

