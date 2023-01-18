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
from utils.data_processing.detect_echoes import (get_envelope,
                                                 get_travel_times,
                                                 find_first_peak)
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

from utils.setups import (Setup,
                          Setup1,
                          Setup2,
                          Setup3)



def setup1_predict_reflections(setup: Setup):
    print('Predicting reflections')
    """Open file"""
    FILE_FOLDER = 'Setup_1/touch'
    FILE_NAME = 'touch_v1'
    measurements = csv_to_df(file_folder=FILE_FOLDER,
                             file_name=FILE_NAME)

    """Interpolate waveforms"""
    measurements = interpolate_waveform(measurements)

    """Compress chirps"""
    # measurements = compress_chirps(measurements)

    """Filter signals by a correlation based bandpass filter"""

    measurements = filter_general(measurements,
                                  filtertype='bandpass',
                                  cutoff_highpass=10000 - 1000,
                                  cutoff_lowpass=10000 + 1000,
                                  order=4,
                                  plot_response=True)
    adjust_plot_margins(['time'])
    file_name = 'bandpass_filter.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name,
                format='pdf')


    setup1_predict_reflections_in_envelopes(setup, measurements)
    # setup1_predict_reflections_in_spectrograms(setup, measurements)


def setup1_predict_reflections_in_envelopes(setup: Setup,
                                            measurements: pd.DataFrame):
    """Calculate arrival times for sensor 1"""
    propagation_speed = setup.get_propagation_speed(measurements)
    print(f'Propagation speed: {propagation_speed}')
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_1],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=True,
                                        print_info=False)
    max_index = np.argmax(np.abs(signal.hilbert(measurements['Sensor 1'])))
    zero_index = measurements.shape[0] / 2
    correction_offset = max_index - zero_index
    correction_offset_time = 2 * correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    """Plot lines for expected arrival times"""
    envelope_with_lines(setup.sensors[SENSOR_1],
                        measurements,
                        arrival_times)
    adjust_plot_margins(['time'])
    file_name = 'predicted_arrivals_envelope_sensor1_setup1.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name,
                format='pdf')

    """Calculate arrival times for sensor 3"""
    arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                        setup.sensors[SENSOR_3],
                                        propagation_speed,
                                        milliseconds=False,
                                        relative_first_reflection=True,
                                        print_info=False)
    max_index = np.argmax(np.abs(signal.hilbert(measurements['Sensor 3'])))
    zero_index = measurements.shape[0] / 2
    correction_offset = max_index - zero_index
    correction_offset_time = 2 * correction_offset / SAMPLE_RATE
    arrival_times += correction_offset_time
    envelope_with_lines(setup.sensors[SENSOR_3],
                        measurements,
                        arrival_times)
    adjust_plot_margins(['time'])
    file_name = 'predicted_arrivals_envelope_sensor3_setup1.pdf'
    plt.savefig(FIGURES_SAVE_PATH + file_name,
                format='pdf')


def make_gaussian_cosine(frequency: float = 1500,
                         num_of_samples: int = SAMPLE_RATE,
                         standard_deviation: float = 1500):
    """Make a cosine of 1000 Hz modulated by a Gaussian pulse"""
    time = np.linspace(-5, 5, num_of_samples)
    effective_frequency = 1 / (1 / frequency)
    cosine = np.cos(2 * np.pi * effective_frequency * time) * \
             signal.gaussian(num_of_samples, std=standard_deviation)

    fig, ax = plt.subplots()
    ax.plot(time, cosine)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_xlim(-0.1, 0.1)
    ax.grid()

    return cosine


if __name__ == '__main__':
    raise ValueError('This script should not be run as main')

