from time import time
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from csv_to_df import csv_to_df
from data_viz_files.visualise_data import crop_data
from data_processing.preprocessing import filter_general
import pandas as pd


def find_indices_of_peaks(sig_np, plot=False):
    """Find the indices of the peaks in the signal."""
    # sig_np = sig.to_numpy()

    # Find the peaks of the square of the signal
    signal_sqr = np.power(sig_np, 2)
    peak_indices_sqr, _ = signal.find_peaks(signal_sqr, prominence=0.0000001) #, distance=0.01 * 150000)

    # Find the peaks of the Hilbert envelope of the signal
    sig_np_filtered_hilbert = get_hilbert_envelope(sig_np)
    peak_indices, _ = signal.find_peaks(sig_np_filtered_hilbert, prominence=0.001, distance=0.01 * 150000)
    if peak_indices.size == 0:
        print('No peaks found!')

    if plot:
        time_axis = np.linspace(0, len(sig_np), num=len(sig_np))
        fig, ax0 = plt.subplots(nrows=1)
        # ax0.plot(time_axis, sig_np, label='signal')
        # ax0.plot(time_axis / 150000, sig_np_filtered_hilbert, label='filtered')
        ax0.plot(time_axis / 150000, signal_sqr, label='sqrd')
        ax0.plot(time_axis[peak_indices_sqr] / 150000, signal_sqr[peak_indices_sqr], 'x', label='peaks')
        ax0.plot(time_axis / 150000, sig_np_filtered_hilbert, label='filtered hilbert')
        ax0.plot(time_axis[peak_indices] / 150000, sig_np_filtered_hilbert[peak_indices], 'x', label='peaks')
        ax0.set_xlabel("Time [s]")
        ax0.legend()
        fig.tight_layout()
        plt.grid()
        plt.show()

    return peak_indices


def get_hilbert_envelope(sig):
    """Get the Hilbert envelope for all channels in df"""
    sig_hilb = sig.copy()
    if isinstance(sig, np.ndarray):
        sig_hilb = np.abs(signal.hilbert(sig))
    elif isinstance(sig, pd.DataFrame):
        for channel in sig_hilb:
            sig_hilb[channel] = np.abs(signal.hilbert(sig[channel]))
        return sig_hilb


def find_first_peak(sig_np, height):
    """Return the index of the first peak of sig_np"""
    peaks, _ = signal.find_peaks(sig_np, height)
    if peaks.size == 0:
        # raise ValueError('No peaks found!')
        return 0
    peak_index = peaks[0]
    return peak_index

def get_expected_reflections_pos(speed, peak,Fs=150000):

    s = [0.26, 0.337, 0.386, 0.41] 
    t = s/speed
    n = t*Fs + peak
    return n.tolist()

if __name__ == '__main__':
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')
    find_indices_of_peaks(signal_df)
