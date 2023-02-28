import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

from utils.plate_setups import Setup
from utils.chirp_gen import generate_chirp_and_save_to_file
from utils.global_constants import (SAMPLE_RATE,
                                    ACTUATOR_1,
                                    SENSOR_1,)
from utils.data_visualization.visualize_data import (
    compare_signals, set_window_size, envelope_with_lines, adjust_plot_margins)
from utils.data_processing.detect_echoes import get_travel_times
from utils.data_processing.preprocessing import (
    compress_chirps, crop_measurement_to_signal)


def generate_test_signal(setup: Setup, propagation_speed_mps: float = 600):
    """Generate a test signal based on expected arrival times for a setup."""

    setup.draw()
    adjust_plot_margins()

    # Generate a chirp for transmission
    TIME_END = 0.125
    FREQUENCY_START = 3_000
    FREQUENCY_STOP = 40_000
    chirp = make_chirp(TIME_END, FREQUENCY_START, FREQUENCY_STOP)

    # Initialize the superpositioned signal
    sensor_measurements = sum_signals(
        setup, propagation_speed_mps, chirp, attenuation_dBpm=20)

    # Compress the test signal
    compressed_test_signal = compress_chirps(sensor_measurements)
    crop_measurement_to_signal(compressed_test_signal, custom_threshold=3000)

    # Plot compressed test signal
    PLOTS_TO_PLOT = ['time', 'fft']
    fig, axs = plt.subplots(4, len(PLOTS_TO_PLOT), squeeze=False)
    compare_signals(fig, axs,
                    data=[compressed_test_signal['Actuator'],
                          compressed_test_signal['Sensor 1'],
                          compressed_test_signal['Sensor 2'],
                          compressed_test_signal['Sensor 3']],
                    compressed_chirps=True,
                    nfft=2**4,
                    plots_to_plot=PLOTS_TO_PLOT,)

    return compressed_test_signal


def make_chirp(TIME_END: float,
               FREQUENCY_START: float,
               FREQUENCY_STOP: float,
               plot_chirp: bool = False):
    chirp = generate_chirp_and_save_to_file(sample_rate=SAMPLE_RATE,
                                            frequency_start=FREQUENCY_START,
                                            frequency_stop=FREQUENCY_STOP,
                                            time_end=TIME_END,
                                            save_to_file=False)

    if plot_chirp:
        # Plot the chirp
        fig, axs = plt.subplots(1, 3, squeeze=False,
                                figsize=set_window_size(rows=1, cols=3))
        compare_signals(fig, axs,
                        data=[chirp],)

    return chirp


def sum_signals(setup: Setup,
                propagation_speed_mps: float,
                chirp: np.ndarray,
                attenuation_dBpm: float = 0,
                plot_signals: bool = False):
    LENGTH_OF_SIGNAL_S = 0.2
    ACTUATOR_CHANNEL = np.pad(
        chirp, (0, int(LENGTH_OF_SIGNAL_S * SAMPLE_RATE)))
    sensor_measurements = pd.DataFrame(data=ACTUATOR_CHANNEL,
                                       columns=['Actuator'])

    # Make signals with the chirp shifted to the expected arrival times
    for sensor_i in range(len(setup.sensors)):
        measurement_i = np.zeros(len(ACTUATOR_CHANNEL))
        arrival_times, _ = get_travel_times(setup.actuators[ACTUATOR_1],
                                            setup.sensors[sensor_i],
                                            propagation_speed_mps,
                                            milliseconds=False,
                                            relative_first_reflection=False,
                                            print_info=False)

        for arrival_time in arrival_times:
            # Calculate the index of the arrival time
            arrival_time_index = int(arrival_time * SAMPLE_RATE)
            travel_distance_m = arrival_time * propagation_speed_mps

            # Add the chirp to the superpositioned signal
            measurement_i[arrival_time_index:arrival_time_index +
                          len(chirp)] += chirp * 10**(-attenuation_dBpm *
                                                      travel_distance_m / 20)

        sensor_measurements[f'Sensor {sensor_i + 1}'] = measurement_i

    # Plot test signal
    if plot_signals:
        PLOTS_TO_PLOT = ['time', 'fft']
        fig, axs = plt.subplots(4, len(PLOTS_TO_PLOT), squeeze=False)
        compare_signals(fig, axs,
                        data=[sensor_measurements['Actuator'],
                              sensor_measurements['Sensor 1'],
                              sensor_measurements['Sensor 2'],
                              sensor_measurements['Sensor 3']],
                        plots_to_plot=PLOTS_TO_PLOT,)

    return sensor_measurements


if __name__ == '__main__':
    raise NotImplementedError
