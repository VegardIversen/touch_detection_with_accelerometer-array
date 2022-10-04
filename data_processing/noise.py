from tkinter.font import names
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\noise_tests\\'
SAMPLE_RATE = 150000     # Hz
DATA_DELIMITER = ","
CHANNEL_NAMES = ['channel 1', 'channel 2', 'channel 3']
FILES = Path(DATA_FOLDER).rglob('*.csv')
OUTPUT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\base_data\\df_average_noise.csv'
Path(OUTPUT_FOLDER).parent.mkdir(parents=True, exist_ok=True)



def SetNoiseAvg(files=FILES, save=True, output_folder=OUTPUT_FOLDER):
    n_files = 0
    for idx, file in enumerate(files):
        print(file.name)
        df = pd.read_csv(file, delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
        if idx == 0:
            df_channel_sum = df
        else:
            df_channel_sum += df 
        n_files += 1

    df_channel_mean = df_channel_sum/n_files 
    print(df_channel_mean.describe())
    df_channel_mean.to_csv(output_folder, index=False)
    return df_channel_mean


if __name__ == '__main__':
    SetNoiseAvg()