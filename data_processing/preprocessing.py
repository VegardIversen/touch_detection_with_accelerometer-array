from scipy import signal
import pandas as pd

import numpy as np
from pathlib import Path

"""FILTERING"""


def filter_general(sig, filtertype, cutoff_lowpass=20000, cutoff_highpass=40000, fs=150000, order=8):
    """filtertype: 'highpass', 'lowpass' or 'bandpass"""
    if filtertype == 'highpass':
        sos = signal.butter(order, cutoff_low / (0.5 * fs), 'highpass', output='sos')
    elif filtertype == 'lowpass':
        sos = signal.butter(order, cutoff_high / (0.5 * fs), 'lowpass',output='sos')
    elif filtertype == 'bandpass':
        sos = signal.butter(order, [cutoff_low / (0.5 * fs),
                             cutoff_high / (0.5 * fs)],
                             'bandpass',
                             output='sos')
    else:
        raise ValueError('Filtertype not recognized')

    sig_filtered = sig.copy()
    if isinstance(sig, pd.DataFrame):
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.sosfilt(sos,
                                                    sig[channel].values)
    else:
        sig_filtered = signal.sosfilt(sos,
                                       sig)

    return sig_filtered


def filter_notches(sig, freqs, sample_rate=150000):
    """Input an array of frequencies <freqs> to filter out
    with a Q factor given by an array of <Qs>.
    """
    for freq in freqs:
        q = freq ** (1 / 3) # We want smaller q-factors for higher frequencies
        b_notch, a_notch = signal.iirnotch(freq / (0.5 * sample_rate), q)
        sig_filtered = sig.copy()
        for channel in sig_filtered:
            # Probably a better way to do this than a double for loop
            sig_filtered[channel] = signal.filtfilt(b_notch, a_notch,
                                                    sig[channel].values)
    return sig_filtered


"""CROPPING"""


def crop_data(sig, time_start=None, time_end=None, threshold=0, sample_rate=150000):
    """Crop either DataFrame input, pandas series or a numpy array input.
    NOTE:   Vegard will fix this to work properly,
            but could use some logic from here.
    """
    # Some logic for assuming cropping type and length
    if (time_start or time_start == 0) and not time_end:
        time_end = len(sig) / sample_rate
    elif time_end and not (time_start or time_start == 0):
        time_start = 0

    if (time_start or time_start ==0) and time_end:
        if isinstance(sig, np.ndarray):
            data_cropped = sig[int(time_start * sample_rate):int(time_end * sample_rate)]
        else:
            # data_cropped = input.loc[time_start * sample_rate:time_end * sample_rate]
            data_cropped = sig.truncate(before=(time_start * sample_rate),
                                        after=(time_end * sample_rate))
    elif not (time_start or time_start == 0) and not time_end:
        if isinstance(sig, pd.DataFrame):
            data_cropped = sig.loc[(sig > threshold).any(axis=1)]
            data_cropped = sig.loc[(sig.iloc[::-1] > threshold).any(axis=1)]
        else:
            data_cropped = sig.loc[sig > threshold]

    return data_cropped


def crop_data_threshold(data, threshold=0.0006):
    if isinstance(data, pd.DataFrame):
        data_cropped = data.loc[(data > threshold).any(axis=1)]
    else:
        data_cropped = data.loc[data > threshold]
    return data_cropped


#create a function that takes in a dataframe and removes silence around the signal
def remove_silence(df, threshold=0.0006):
    df = df.loc[(df > threshold).any(axis=1)]
    #reverse df
    rev = df.iloc[::-1].reset_index(drop=True)
    #remove silence from the end_fre
    rev = rev.loc[(rev > threshold).any(axis=1)]
    #reverse again
    df_cleand = rev.iloc[::-1].reset_index(drop=True)
    return df_cleand
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ROOT_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'
    file_folder='div_files/setup3'
    file_name='sign_integ_test_touchhold_fulldurr_150k_5s_setup3_1_v1'
    file_path = ROOT_FOLDER + '\\' + file_folder + '\\' + file_name + '.csv'
    channel_names = ['channel 1', 'channel 2', 'channel 3', 'WAVE_GEN']
    test = pd.read_csv(filepath_or_buffer=file_path, names=channel_names)
    test = test.drop(columns=['WAVE_GEN'])
    test_removed_silence = remove_silence(test)
    ch1 = test['channel 1'].to_numpy()
    rev = test[::-1].reset_index(drop=True)
    plt.subplot(211)
    plt.plot(test_removed_silence)
    plt.subplot(212)
    plt.plot(test)
    plt.show()
