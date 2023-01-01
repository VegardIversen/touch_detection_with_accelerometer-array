import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from pathlib import Path
from data_processing.preprocessing import get_first_index_above_threshold, compress_single_touch, compress_chirp, manual_cut_signal


def phase_difference_sub(sig1, sig2):
    """
    Calculates the phase difference between two signals using the FFT method.
    :param sig1: First signal
    :param sig2: Second signal
    :return: Phase difference between the two signals
    """
    sig1_fft = np.fft.fft(sig1)
    sig2_fft = np.fft.fft(sig2)
    sig1_fft = np.fft.fftshift(sig1_fft)
    sig2_fft = np.fft.fftshift(sig2_fft)
    sig1_fft = np.abs(sig1_fft)
    sig2_fft = np.abs(sig2_fft)
    sig1_fft = sig1_fft/np.max(sig1_fft)
    sig2_fft = sig2_fft/np.max(sig2_fft)
    sig1_fft = np.log(sig1_fft)
    sig2_fft = np.log(sig2_fft)
    sig1_fft = np.unwrap(np.angle(sig1_fft))
    sig2_fft = np.unwrap(np.angle(sig2_fft))
    phase_diff = sig1_fft - sig2_fft
    return phase_diff

def phase_difference_div(sig1, sig2, n_pi=0):
    S1f = np.fft.fft(sig1)
    S2f = np.fft.fft(sig2)
    phase = np.unwrap(np.angle(S2f/S1f)) +n_pi*np.pi
    return phase

def phase_difference_gpt(sig1, sig2):
    fft_sig1 = np.fft.fft(sig1)
    fft_sig2 = np.fft.fft(sig2)

    # Find the phase angles of the FFTs of the two pulse compressed signals
    phase_angles_sig1 = np.angle(fft_sig1)
    phase_angles_sig2 = np.angle(fft_sig2)

    # Use the unwrap function to remove any jumps in phase between consecutive elements
    unwrapped_phase_angles_sig1 = np.unwrap(phase_angles_sig1)
    unwrapped_phase_angles_sig2 = np.unwrap(phase_angles_sig2)

    # Calculate the phase difference between the two pulse compressed signals
    phi = unwrapped_phase_angles_sig2 - unwrapped_phase_angles_sig1

    return phi

def phase_difference(sig1, sig2, method='sub', n_pi=0):
    if method == 'sub':
        return phase_difference_sub(sig1, sig2)
    elif method == 'div':
        return phase_difference_div(sig1, sig2, n_pi=n_pi)
    elif method == 'gpt':
        return phase_difference_gpt(sig1, sig2)
    else:
        raise ValueError('Invalid method')

def phase_difference_plot(sig1, sig2, method='div', n_pi=0, SAMPLE_RATE=150000, BANDWIDTH=None):
    phase_diff = phase_difference(sig1, sig2, method=method, n_pi=n_pi)
    freq = np.fft.fftfreq(len(sig1), 1/SAMPLE_RATE)
    freq = np.fft.fftshift(freq)
    if BANDWIDTH is not None:
        slices = (freq>BANDWIDTH[0]) & (freq<BANDWIDTH[1])
        phase_diff = phase_diff[slices]
        freq = freq[slices]
    plt.plot(freq, phase_diff)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase difference [rad]')
    plt.show()
    
def preprocess_df(df, detrend=True):
    if detrend:
        for col in df.columns:
            df[col] = signal.detrend(df[col])
    return df

def compress_and_cut_df_touch(df, channels=['channel 1', 'channel 3'], direct_samples=55):
    compressed_df = df.copy()
    for ch in channels:
        compressed_df[ch] = compress_single_touch(df[ch], set_threshold_man=True, n_sampl=direct_samples)
    return compressed_df

def plot_results(df, channels=['channel 1', 'channel 3'], direct_samples=55, detrend=True, method='gpt', n_pi=0, SAMPLE_RATE=150000, BANDWIDTH=None, chirp=None):
    df = preprocess_df(df, detrend=detrend)
    if chirp is not None:
        df = compress_chirp(df, chirp)
    df = compress_and_cut_df_touch(df, channels=channels, direct_samples=direct_samples)
    phase_difference_plot(df[channels[0]], df[channels[1]], method=method, n_pi=n_pi, SAMPLE_RATE=SAMPLE_RATE, BANDWIDTH=BANDWIDTH)

def phase_plotting(df, channels=['channel 1', 'channel 3'], detrend=True):
    start1,end1 = manual_cut_signal(signal=df[channels[0]])
    start2,end2 = manual_cut_signal(signal=df[channels[1]])
    #empty array to fill
    temp_arr = np.zeros((len(df[channels[0]]),len(channels)))
    #filling the array
    temp_arr[start1:end1,0] = df[channels[0]].iloc[start1:end1]
    temp_arr[start2:end2,1] = df[channels[1]].iloc[start2:end2]
    df_zero = pd.DataFrame(temp_arr, columns=channels)
    