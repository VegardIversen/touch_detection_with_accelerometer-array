from cProfile import label
import pandas as pd
import numpy as np
from pathlib import Path
import padasip as pa
import matplotlib.pyplot as plt
import noisereduce as nr

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


def adaptive_filter_RLS(signal, n=20, mu=0.9):
    x = pa.input_from_history(signal, n)[:-1]
    sig = signal[n:]
    f = pa.filters.FilterRLS(mu=mu, n=n)
    y, e, w = f.run(sig, x)
    return y, e, w


def adaptive_filter_NLMS(signal, n=10, mu=0.25):
    x = pa.input_from_history(signal, n)[:-1]
    sig = signal[n:]
    f = pa.filters.FilterNLMS(mu=mu, n=n)
    y, e, w = f.run(sig, x)
    return y, e, w


def noise_reduce_signal(sig, noise, show=False):
    reduced_noise = nr.reduce_noise(y=sig, sr=SAMPLE_RATE, y_noise=noise)
    if show:

        ax1 = plt.subplot(211)
        plt.plot(sig)
        plt.subplot(212, sharex=ax1, sharey=ax1)
        plt.plot(reduced_noise)
        plt.show()
    return reduced_noise


if __name__ == '__main__':
    #SetNoiseAvg()
    noise = pd.read_csv(OUTPUT_FOLDER, delimiter=DATA_DELIMITER)
    data = pd.read_csv(f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_A1_center_v1.csv', delimiter=DATA_DELIMITER, names=CHANNEL_NAMES)
    #adaptive_filter_RLS(data['channel 1'])
    #noise_reduce_signal(data['channel 1'], noise['channel 1'])