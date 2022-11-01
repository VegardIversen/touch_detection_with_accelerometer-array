import pandas as pd
from pathlib import Path
from sys import platform


def csv_to_df(file_folder, file_name, channel_names=['channel 1', 'channel 2', 'channel 3']):
    """Returns a DataFrame from a .csv file.
    Set channel_names to None to return a
    DataFrame with the default column names.
    """
    if platform == 'linux':
        ROOT_FOLDER = '/mnt/c/Users/nikla/OneDrive - NTNU/NTNU/ProsjektOppgave'
        file_path = ROOT_FOLDER + '/' + file_folder + '/' + file_name + '.csv'
    else:
        ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
        file_path = ROOT_FOLDER + '\\' + file_folder + '\\' + file_name + '.csv'

    df = pd.read_csv(filepath_or_buffer=file_path, names=channel_names)
    print("\n" + 'Creating DataFrame from file folder "' + file_folder + '" and file name "' + file_name + '"\n')

    return df


if __name__ == '__main__':
    csv_to_df("base_data", "df_average_noise")