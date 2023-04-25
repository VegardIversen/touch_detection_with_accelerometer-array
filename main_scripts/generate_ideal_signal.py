import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from utils.csv_to_df import make_dataframe_from_csv

from utils.data_processing.detect_echoes import get_envelopes, get_travel_times
from utils.data_processing.preprocessing import (
    compress_chirps,
    crop_data,
    crop_to_signal,
    filter_signal,
)
from utils.data_processing.processing import (
    align_signals_by_max_value,
    interpolate_signal,
    normalize,
)
from utils.data_visualization.visualize_data import (
    compare_signals,
    set_window_size,
)
from utils.generate_chirp import generate_chirp
from utils.global_constants import (
    ACTUATOR_1,
    ORIGINAL_SAMPLE_RATE,
    SAMPLE_RATE,
)
from utils.plate_setups import Setup


def compare_to_ideal_signal(
    setup: Setup,
    measurements: pd.DataFrame,
    attenuation_dBpm: float,
    propagation_speed_mps: float = None,
    cutoff_frequency: float = 0,
):
    """Calculate arrival times for sensor 1"""
    if propagation_speed_mps is None:
        propagation_speed_mps = 950
    print(f"Propagation speed: {propagation_speed_mps:.2f}")
    signal_length_s = float(
        measurements.shape[0] / SAMPLE_RATE,
    )
    ideal_signal, distances = generate_ideal_signal(
        setup,
        propagation_speed_mps,
        attenuation_dBpm,
        signal_length_s,
        cutoff_frequency,
    )
    measurement_envelopes = get_envelopes(measurements)
    measurement_envelopes = normalize(measurement_envelopes)
    ideal_signal = align_signals_by_max_value(
        signals=ideal_signal, signals_to_align_with=measurement_envelopes
    )
    """Plot signals"""
    CHANNELS_TO_PLOT = setup.sensors
    PLOTS_TO_PLOT = ["time"]
    fig, axs = plt.subplots(
        nrows=len(CHANNELS_TO_PLOT), ncols=len(PLOTS_TO_PLOT), squeeze=False
    )
    compare_signals(
        fig,
        axs,
        [
            get_envelopes(ideal_signal["Sensor 1"]),
            get_envelopes(ideal_signal["Sensor 2"]),
            get_envelopes(ideal_signal["Sensor 3"]),
        ],
        plots_to_plot=PLOTS_TO_PLOT,
    )
    compare_signals(
        fig,
        axs,
        [
            measurement_envelopes["Sensor 1"],
            measurement_envelopes["Sensor 2"],
            measurement_envelopes["Sensor 3"],
        ],
        plots_to_plot=PLOTS_TO_PLOT,
    )
    [ax.grid() for ax in axs[:, 0]]
    axs[0, 0].legend(["Ideal signal", "Measurement envelope"], loc="upper right")
    fig.suptitle("Ideal signal vs. measurement envelope")
    return ideal_signal, distances


def generate_ideal_signal(
    setup: Setup,
    propagation_speed_mps: float,
    attenuation_dBpm: float,
    signal_length_s: float,
    cutoff_frequency: float = 0,
    signal_model: str = "touch",
):
    """Generate an "ideal" signal based on expected arrival times for a setup."""
    if signal_model == "touch":
        touch_signal = extract_touch_signal(filter_critical_frequency=cutoff_frequency)
    elif signal_model == "line":
        # Generate a dirac pulse
        touch_signal = np.zeros(int(signal_length_s * SAMPLE_RATE))
        touch_signal[int(signal_length_s * SAMPLE_RATE / 2)] = 1

    # Initialize the superpositioned signal
    sensor_measurements, distances = sum_signals(
        setup,
        propagation_speed_mps,
        touch_signal,
        attenuation_dBpm,
        signal_length_s,
    )

    # Compress the test signal
    ideal_signal = compress_chirps(sensor_measurements)
    return ideal_signal, distances


def make_chirp(
    time_end: float,
    frequency_start: float,
    frequency_stop: float,
    plot_chirp: bool = False,
):
    """An alternative to using the touch signal for the ideal signal"""
    chirp = generate_chirp(
        sample_rate=SAMPLE_RATE,
        frequency_start=frequency_start,
        frequency_stop=frequency_stop,
        time_end_s=time_end,
        save_to_file=False,
    )
    if plot_chirp:
        fig, axs = plt.subplots(
            1, 3, squeeze=False, figsize=set_window_size(rows=1, cols=3)
        )
        compare_signals(
            fig,
            axs,
            measurements=[chirp],
        )
        fig.suptitle("The generated chirp")
    return chirp


def sum_signals(
    setup: Setup,
    propagation_speed_mps: float,
    touch_signal: np.ndarray,
    attenuation_dBpm: float = 0,
    signal_length_s: float = 5,
    plot_signals: bool = True,
):
    ACTUATOR_CHANNEL = np.pad(touch_signal, (0, int(signal_length_s * SAMPLE_RATE)))
    sensor_measurements = pd.DataFrame(data=ACTUATOR_CHANNEL, columns=["Actuator"])
    travel_distances = []

    for sensor_i in range(len(setup.sensors)):
        measurement_i = np.zeros(len(ACTUATOR_CHANNEL))
        arrival_times, distances = get_travel_times(
            setup.actuators[ACTUATOR_1],
            setup.sensors[sensor_i],
            propagation_speed_mps,
            surface=setup.surface,
            milliseconds=False,
            relative_first_reflection=False,
            print_info=False,
        )
        for arrival_time in arrival_times:
            arrival_time_index = int(arrival_time * SAMPLE_RATE)
            travel_distance_m = arrival_time * propagation_speed_mps
            measurement_i[
                arrival_time_index : arrival_time_index + len(touch_signal)
            ] += touch_signal * 10 ** (-attenuation_dBpm * travel_distance_m / 20)
        sensor_measurements[f"Sensor {sensor_i + 1}"] = measurement_i
        travel_distances.append(distances[:2])

    if plot_signals:
        PLOTS_TO_PLOT = ["time"]
        fig, axs = plt.subplots(9, len(PLOTS_TO_PLOT), squeeze=False)
        compare_signals(
            fig,
            axs,
            measurements=[
                sensor_measurements[sensor_i]
                for sensor_i in sensor_measurements.columns
            ],
            plots_to_plot=PLOTS_TO_PLOT,
            sharey=True,
        )
        # Set the y-axis labels
        for ax in axs[:, 0]:
            ax.set_ylabel(f"Sensor {axs[:, 0].tolist().index(ax)}")
        # Set the first y-axis label to touch
        axs[0, 0].set_ylabel("Touch")
    return sensor_measurements, travel_distances


def extract_touch_signal(
    filter_critical_frequency: float = 0,
    plot_signal: bool = True,
    interpolate: bool = True,
    window: bool = True,
):
    FILE_FOLDER = "Plate_10mm/Setup3/touch"
    FILE_NAME = "nik_touch_v1"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    measurements = crop_to_signal(measurements)
    measurements = crop_data(
        measurements,
        time_start=0.05645,
        time_end=0.05667,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    touch_signal = measurements["Sensor 1"].values

    if filter_critical_frequency:
        touch_signal = filter_signal(
            touch_signal,
            filtertype="highpass",
            critical_frequency=filter_critical_frequency,
        )

    if interpolate:
        touch_signal = interpolate_signal(touch_signal)

    if window:
        touch_signal = signal.windows.hann(len(touch_signal)) * touch_signal

    if plot_signal:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            squeeze=False,
        )
        compare_signals(
            fig,
            axs,
            measurements=[touch_signal],
            nfft=2**7,
        )
        fig.suptitle("Touch signal used for ideal measurement", fontsize=16)

    return touch_signal


if __name__ == "__main__":
    raise NotImplementedError
