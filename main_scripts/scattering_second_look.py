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

from utils.table_setups import (Setup,
                                Setup1,
                                Setup2,
                                Setup3)


def scattering_second_look():
    FILE_FOLDER = 'Table/Setup1/scattering_tests/15kHz_to_40kHz_125ms'
    FILE_NAME = 'no_touch_v1'
    # FILE_FOLDER = 'Table\\Setup3'
    # FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    no_touch = csv_to_df(file_folder=FILE_FOLDER,
                         file_name=FILE_NAME)

    no_touch = interpolate_waveform(no_touch)

    no_touch = filter_general(no_touch,
                              filtertype='highpass',
                              critical_frequency=1,
                              order=8,
                              plot_response=False,)

    FILE_FOLDER = 'Table/Setup1/scattering_tests/15kHz_to_40kHz_125ms'
    FILE_NAME = 'hold_pos_2_v1'
    # FILE_FOLDER = 'Table\\Setup3'
    # FILE_NAME = 'notouchThenHoldB2_20to40khz_125ms_10vpp_v1'
    touch = csv_to_df(file_folder=FILE_FOLDER,
                      file_name=FILE_NAME)

    touch = interpolate_waveform(touch)

    touch = filter_general(touch,
                           filtertype='highpass',
                           critical_frequency=1,
                           order=8,
                           plot_response=False,)

    """Shift to align chirps better with their time intervals"""
    # SHIFT_BY = int(0.0877 * SAMPLE_RATE)
    # for channel in no_touch:
    #     no_touch[channel] = np.roll(no_touch[channel],
    #                                 -SHIFT_BY)
    #     no_touch[channel][-SHIFT_BY:] = 0

    """Compress chirps"""
    CHIRP_LENGTH = int(0.125 * SAMPLE_RATE)
    PRE_PAD_LENGTH = int(2.5 * SAMPLE_RATE)
    POST_PAD_LENGTH = no_touch.shape[0] - \
        (PRE_PAD_LENGTH + CHIRP_LENGTH) - 1
    chirp_no_touch = no_touch['Actuator'][0:CHIRP_LENGTH + 1]
    chirp_touch = touch['Actuator'][0:CHIRP_LENGTH + 1]
    reference_no_touch = np.pad(chirp_no_touch,
                                (PRE_PAD_LENGTH, POST_PAD_LENGTH),
                                mode='constant')
    reference_touch = np.pad(chirp_touch,
                             (PRE_PAD_LENGTH, POST_PAD_LENGTH),
                             mode='constant')
    for channel in no_touch:
        no_touch[channel] = signal.correlate(no_touch[channel],
                                             reference_no_touch,
                                             mode='same')
    for channel in touch:
        touch[channel] = signal.correlate(touch[channel],
                                          reference_touch,
                                          mode='same')

    """Plot measurements"""
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=2,
                            ncols=len(PLOTS_TO_PLOT),
                            figsize=set_window_size(),
                            squeeze=False)
    diff_signal_1 = no_touch['Sensor 1'] - touch['Sensor 1']
    diff_envelope_1 = np.abs(signal.hilbert(diff_signal_1))
    diff_signal_2 = no_touch['Sensor 3'] - touch['Sensor 3']
    diff_envelope_2 = np.abs(signal.hilbert(diff_signal_2))
    compare_signals(fig, axs,
                    data=[normalize(np.abs(signal.hilbert(no_touch['Sensor 1']))),
                          normalize(np.abs(signal.hilbert(no_touch['Sensor 3'])))],
                    plots_to_plot=PLOTS_TO_PLOT,)
    compare_signals(fig, axs,
                    data=[normalize(np.abs(signal.hilbert(touch['Sensor 1']))),
                          normalize(np.abs(signal.hilbert(touch['Sensor 3'])))],
                    plots_to_plot=PLOTS_TO_PLOT,)
    compare_signals(fig, axs,
                    data=[10 * normalize(diff_envelope_1),
                          10 * normalize(diff_envelope_2)],
                    plots_to_plot=PLOTS_TO_PLOT,)

    setup = Setup1()
    propagation_speed = setup.get_propagation_speed(no_touch)
    print(f'Propagation speed: {propagation_speed:.2f} m/s')

