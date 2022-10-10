from symbol import file_input
from threading import TIMEOUT_MAX
from time import time
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from csv_to_df import csv_to_df
from data_viz_files.visualise_data import crop_data
from data_processing.preprocessing import high_pass_filter, low_pass_filter


def find_indices_of_peaks(sig, plot=False):
    """Find the indices of the peaks in the signal."""
    sig_np = sig.to_numpy()
    # Crop the signal
    sig_np_filtered = high_pass_filter(sig_np, cutoff=1000, order=5)
    sig_np_filtered_hilbert = get_hilbert_envelope(sig_np_filtered)
    # peak_indices = signal.argrelmax(sig_np_filtered_hilbert, order=10000)[0]
    # peak_indices = signal.find_peaks(sig_np_filtered_hilbert, height=0.0005, threshold=0.0001)[0]
    peak_indices = signal.find_peaks_cwt(sig_np_filtered_hilbert, widths=5)
    if peak_indices.size == 0:
        print('No peaks found!')

    if plot:
        time_axis = np.linspace(0, len(sig_np), num=len(sig_np))
        fig, ax0 = plt.subplots(nrows=1)
        # ax0.plot(time_axis, sig_np, label='signal')
        ax0.plot(time_axis / 150000, sig_np_filtered, label='filtered')
        ax0.plot(time_axis / 150000, sig_np_filtered_hilbert, label='filtered hilbert')
        ax0.plot(time_axis[peak_indices] / 150000, sig_np_filtered_hilbert[peak_indices], 'x', label='peaks')
        ax0.set_xlabel("Time [s]")
        ax0.legend()
        fig.tight_layout()
        plt.grid()
        plt.show()

    return peak_indices


def get_hilbert_envelope(sig_np):
    analytic_signal = signal.hilbert(sig_np)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


if __name__ == '__main__':
    signal_df = csv_to_df(file_folder='first_test_touch_passive_setup2',
                          file_name='touch_test_passive_setup2_place_C3_center_v2')
    find_indices_of_peaks(signal_df)
