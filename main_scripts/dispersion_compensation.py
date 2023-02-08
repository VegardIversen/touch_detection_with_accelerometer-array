import scipy
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.global_constants import (CHIRP_CHANNEL_NAMES,
                                    SAMPLE_RATE,
                                    ACTUATOR_1,
                                    SENSOR_1,
                                    SENSOR_2,
                                    SENSOR_3,
                                    FIGURES_SAVE_PATH)
from utils.csv_to_df import csv_to_df
from utils.simulations import simulated_phase_velocities
from utils.data_processing.detect_echoes import (get_envelopes,
                                                 get_travel_times,
                                                 find_first_peak_index)
from utils.data_processing.preprocessing import (compress_chirps,
                                                 crop_data,
                                                 window_signals,
                                                 filter_general)
from utils.data_processing.processing import (average_of_signals,
                                              interpolate_waveform,
                                              normalize,
                                              variance_of_signals,
                                              correct_drift,
                                              to_dB)
from utils.data_visualization.visualize_data import (compare_signals,
                                                     wave_statistics,
                                                     envelope_with_lines,
                                                     spectrogram_with_lines,
                                                     set_window_size,
                                                     adjust_plot_margins)
from main_scripts.correlation_bandpassing import (make_gaussian_cosine)

from utils.setups import (Setup,
                          Setup1,
                          Setup2,
                          Setup3)


def dispersive_filter():
    # Define the length of the signal in seconds
    LENGTH = 0.1
    SAMPLE_RATE = 1e7

    # Calculate the number of samples
    N = int(SAMPLE_RATE * LENGTH)

    # Generate a chirp with bandwidth 40 kHz
    chirp = signal.chirp(np.linspace(0, LENGTH, N), 0, LENGTH, 40000)
    impulse = signal.correlate(chirp, chirp, mode='same')

    # Calculate the FFT
    impulse_fft = np.fft.fft(impulse)
    impulse_fft_frequencies = np.fft.fftfreq(N, 1 / SAMPLE_RATE)
    # Crop to 40 kHz
    # impulse_fft = impulse_fft[impulse_fft_frequencies < 40000]
    # impulse_fft_frequencies = impulse_fft_frequencies[impulse_fft_frequencies < 40000]

    # A linear graph providing the phase offset
    phase_offset = np.linspace(0.05, -0.05, len(impulse_fft))
    new_chirp = np.fft.ifft(impulse_fft * np.exp(-1j * phase_offset * impulse_fft_frequencies))

    # Calculate the time values for each sample
    time = np.linspace(0, LENGTH, N)

    # Plot the signal
    _, ax = plt.subplots()
    ax.plot(time, impulse, label='Original')
    ax.plot(time, new_chirp, label='New chirp')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid()

    plt.show()

    return 0
