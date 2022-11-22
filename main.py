import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

from constants import CHIRP_CHANNEL_NAMES, SAMPLE_RATE
from csv_to_df import csv_to_df
from data_processing.detect_echoes import (get_hilbert_envelope,
                                           get_travel_times)
from data_processing.preprocessing import (compress_chirp,
                                           crop_data,
                                           filter_general)
from data_processing.processing import (avg_waveform,
                                        interpolate_waveform,
                                        normalize,
                                        var_waveform,
                                        correct_drift)
from data_viz_files.visualise_data import (compare_signals,
                                           wave_statistics,
                                           set_fontsizes)
from objects import Table
from setups import Setup7


def main():
    """CONFIG"""
    FILE_FOLDER = 'setup7'
    FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    SETUP = Setup7()
    CROP = False
    TIME_START = 0.114  # s
    TIME_END = 0.246  # s
    BANDWIDTH = (200, 40000)

    set_fontsizes()

    """Open file"""
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME,
                             channel_names=CHIRP_CHANNEL_NAMES)

    """Crop data"""
    measurements = crop_data(measurements,
                             time_start=TIME_START,
                             time_end=TIME_END)

    """Plot raw signal"""
    time_axis = np.linspace(start=0,
                            stop=1000 * len(measurements) / SAMPLE_RATE,
                            num=len(measurements))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    plt.suptitle('Chirp from 20 khz to 40 kHz in 125 ms')
    axs[0].plot(time_axis, 1000 * measurements['Sensor 1'], label='Sensor 1')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Amplitude (mV)')
    axs[0].legend()
    axs[0].grid()
    """FFT of raw signal"""
    freq_axis, fft = signal.welch(measurements['Sensor 1'],
                                  fs=SAMPLE_RATE,
                                  nperseg=2**12)
    axs[1].plot(freq_axis / 1000, 10 * np.log10(fft), label='Sensor 1')
    axs[1].set_xlabel('Frequency (kHz)')
    axs[1].set_ylabel('Amplitude (dB)')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
