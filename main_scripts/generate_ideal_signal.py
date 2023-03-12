import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils.data_processing.processing import align_signals_by_max_value, normalize

from utils.plate_setups import Setup
from utils.generate_chirp import generate_chirp
from utils.global_constants import (SAMPLE_RATE,
                                    ACTUATOR_1,
                                    SENSOR_1,)
from utils.data_visualization.visualize_data import (compare_signals,
                                                     set_window_size,
                                                     adjust_plot_margins)
from utils.data_processing.detect_echoes import get_envelopes, get_travel_times
from utils.data_processing.preprocessing import (compress_chirps)


def compare_to_ideal_signal(setup: Setup,
                            measurements: pd.DataFrame,
                            attenuation_dBpm: float,
                            chirp_length_s: float,
                            frequency_start: float,
                            frequency_stop: float,
                            propagation_speed_mps: float = None,):
    """Calculate arrival times for sensor 1"""
    if propagation_speed_mps is None:
        # propagation_speed = setup.get_propagation_speed(measurements)
        propagation_speed_mps = 897.27
    print(f'Propagation speed: {propagation_speed_mps:.2f}')
    signal_length_s = float(measurements.shape[0] / SAMPLE_RATE,)
    ideal_signal, distances = generate_ideal_signal(setup,
                                                    propagation_speed_mps,
                                                    attenuation_dBpm,
                                                    chirp_length_s,
                                                    frequency_start,
                                                    frequency_stop,
                                                    signal_length_s)
    measurement_envelopes = get_envelopes(measurements)
    measurement_envelopes = normalize(measurement_envelopes)
    ideal_signal = align_signals_by_max_value(
        signals=ideal_signal, signals_to_align_with=measurement_envelopes)
    """Plot signals"""
    CHANNELS_TO_PLOT = setup.sensors
    PLOTS_TO_PLOT = ['time']
    fig, axs = plt.subplots(nrows=len(CHANNELS_TO_PLOT),
                            ncols=len(PLOTS_TO_PLOT),
                            squeeze=False)
    compare_signals(fig, axs,
                    [ideal_signal['Sensor 1'],
                     ideal_signal['Sensor 2'],
                     ideal_signal['Sensor 3']],
                    plots_to_plot=PLOTS_TO_PLOT)
    compare_signals(fig, axs,
                    [measurement_envelopes['Sensor 1'],
                     measurement_envelopes['Sensor 2'],
                     measurement_envelopes['Sensor 3'],],
                    plots_to_plot=PLOTS_TO_PLOT)
    [ax.grid() for ax in axs[:, 0]]
    axs[0, 0].legend(['Ideal signal', 'Measurement envelope'],
                     loc='upper right')
    return ideal_signal, distances


def generate_ideal_signal(setup: Setup,
                          propagation_speed_mps: float,
                          attenuation_dBpm: float,
                          chirp_length_s: float,
                          frequency_start: float,
                          frequency_stop: float,
                          signal_length_s: float):
    """Generate a test signal based on expected arrival times for a setup."""
    # Generate a chirp for transmission
    chirp = make_chirp(chirp_length_s, frequency_start, frequency_stop)

    # Initialize the superpositioned signal
    sensor_measurements, distances = sum_signals(setup,
                                                 propagation_speed_mps,
                                                 chirp,
                                                 attenuation_dBpm,
                                                 signal_length_s,)

    # Compress the test signal
    compressed_ideal_signal = compress_chirps(sensor_measurements)
    # crop_measurement_to_signal_ndarray(compressed_ideal_signal)
    return compressed_ideal_signal, distances


def make_chirp(time_end: float,
               frequency_start: float,
               frequency_stop: float,
               plot_chirp: bool = False):
    chirp = generate_chirp(sample_rate=SAMPLE_RATE,
                           frequency_start=frequency_start,
                           frequency_stop=frequency_stop,
                           time_end_s=time_end,
                           save_to_file=False)
    if plot_chirp:
        fig, axs = plt.subplots(1, 3, squeeze=False,
                                figsize=set_window_size(rows=1, cols=3))
        compare_signals(fig, axs,
                        data=[chirp],)
    return chirp


def sum_signals(setup: Setup,
                propagation_speed_mps: float,
                chirp: np.ndarray,
                attenuation_dBpm: float = 0,
                signal_length_s: float = 5,
                plot_signals: bool = False,):
    ACTUATOR_CHANNEL = np.pad(chirp, (0, int(signal_length_s * SAMPLE_RATE)))
    sensor_measurements = pd.DataFrame(data=ACTUATOR_CHANNEL,
                                       columns=['Actuator'])
    travel_distances = []

    for sensor_i in range(len(setup.sensors)):
        measurement_i = np.zeros(len(ACTUATOR_CHANNEL))
        arrival_times, distances = get_travel_times(setup.actuators[ACTUATOR_1],
                                                    setup.sensors[sensor_i],
                                                    propagation_speed_mps,
                                                    surface=setup.surface,
                                                    milliseconds=False,
                                                    relative_first_reflection=False,
                                                    print_info=False)
        for arrival_time in arrival_times:
            arrival_time_index = int(arrival_time * SAMPLE_RATE)
            travel_distance_m = arrival_time * propagation_speed_mps
            measurement_i[arrival_time_index:arrival_time_index +
                          len(chirp)] += chirp * 10**(-attenuation_dBpm *
                                                      travel_distance_m / 20)
        sensor_measurements[f'Sensor {sensor_i + 1}'] = measurement_i
        travel_distances.append(distances[:2])
        print(
            f'First two distances for sensor {sensor_i + 1}: {distances[:2]}')

    if plot_signals:
        PLOTS_TO_PLOT = ['time', 'fft']
        fig, axs = plt.subplots(4, len(PLOTS_TO_PLOT), squeeze=False)
        compare_signals(fig, axs,
                        data=[sensor_measurements['Actuator'],
                              sensor_measurements['Sensor 1'],
                              sensor_measurements['Sensor 2'],
                              sensor_measurements['Sensor 3']],
                        plots_to_plot=PLOTS_TO_PLOT,)
    return sensor_measurements, travel_distances


if __name__ == '__main__':
    raise NotImplementedError
