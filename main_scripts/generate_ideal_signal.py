import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from utils.csv_to_df import make_dataframe_from_csv

from utils.data_processing.detect_echoes import get_envelopes, get_travel_times
from utils.data_processing.preprocessing import (
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
    filtertype: str = "highpass",
    critical_frequency: float = 0,
    signal_model: str = "touch",
):
    """Calculate arrival times for sensor 1"""
    if propagation_speed_mps is None:
        propagation_speed_mps = 922
    print(f"Propagation speed: {propagation_speed_mps:.2f}")
    signal_length_s = float(
        measurements.shape[0] / SAMPLE_RATE,
    )
    ideal_signal, distances = generate_ideal_signal(
        setup,
        propagation_speed_mps,
        attenuation_dBpm,
        signal_length_s,
        critical_frequency,
        signal_model,
    )
    measurements = filter_signal(
        measurements,
        filtertype=filtertype,
        critical_frequency=critical_frequency,
        plot_response=False,
        order=2,
        sample_rate=SAMPLE_RATE,
        q=0.05,
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
    group_velocity_mps: float,
    phase_velocity_mps: float,
    attenuation_dBpm: float,
    signal_length_s: float,
    signal_model: str = "line",
    center_frequency_Hz: float = 0,
    t_var: float = 1e-9,
    snr_dB: float = 0,
):
    """Generate an "ideal" signal based on expected arrival times for a setup."""
    touch_signal = model_touch_signal(
        signal_length_s,
        signal_model,
        center_frequency_Hz,
        t_var,
    )

    # Initialize the superpositioned signal
    ideal_signals, distances = sum_signals(
        setup,
        group_velocity_mps,
        touch_signal,
        attenuation_dBpm,
        signal_length_s,
    )

    ideal_signals = add_noise(
        ideal_signals,
        center_frequency_Hz,
        snr_dB,
    )
    return ideal_signals, distances


def model_touch_signal(
    signal_length_s,
    signal_model,
    critical_frequency,
    t_var,
):
    if signal_model == "touch":
        touch_signal = extract_touch_signal(
            filter_critical_frequency=critical_frequency
        )
    elif signal_model == "line":
        # Generate a dirac pulse
        touch_signal = np.zeros(int(signal_length_s * SAMPLE_RATE))
        touch_signal[int(signal_length_s * SAMPLE_RATE / 2)] = 1
    elif signal_model == "gaussian":
        touch_signal = model_gaussian_touch(
            signal_length_s,
            critical_frequency,
            t_var,
        )
    else:
        raise ValueError("Invalid signal model")
    return touch_signal


def model_gaussian_touch(signal_length_s, critical_frequency, t_var):
    # Generate a gaussian modulated pulse with frequency critical_frequency and duration 10 periods
    t = np.linspace(
        -signal_length_s / 2,
        signal_length_s / 2,
        int(signal_length_s * SAMPLE_RATE),
    )
    # The function Tonni uses for the simulations
    touch_signal = -np.exp(-((t) ** 2) / (2 * t_var)) * np.sin(
        2 * np.pi * critical_frequency * (t)
    )
    # Add some noise to allow crop_to_signal() to work properly
    touch_signal_with_noise = add_noise(touch_signal, critical_frequency, snr_dB=50)
    touch_signal_with_noise, _, _ = crop_to_signal(touch_signal_with_noise)
    time_axis_for_plotting = np.linspace(
        0, len(touch_signal_with_noise) / SAMPLE_RATE, len(touch_signal_with_noise)
    )
    # Plot the signal
    fig, ax = plt.subplots()
    ax.plot(time_axis_for_plotting, touch_signal_with_noise)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Gaussian modulated pulse")
    return touch_signal


def add_noise(
    ideal_signals: pd.DataFrame or np.ndarray,
    critical_frequency: float,
    snr_dB: float = 0,
):
    if isinstance(ideal_signals, pd.DataFrame):
        # Call the function recursively for each channel
        for channel in ideal_signals.columns:
            ideal_signals[channel] = add_noise(
                ideal_signals[channel],
                critical_frequency,
                snr_dB,
            )
        return ideal_signals

    ideal_signal_power = np.sum(ideal_signals**2) / len(ideal_signals)
    noise_power = ideal_signal_power / (10 ** (snr_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(ideal_signals))
    noisy_signal = ideal_signals + noise
    return noisy_signal


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
    plot_signals: bool = False,
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
        # Hardcode the arrival times to be only the indices 0, 3, 4, and 11 of arrival_times
        # arrival_times = arrival_times[[0, 3, 4, 11]]
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
        fig.suptitle("Touch signal used for ideal measurement")

    return touch_signal


if __name__ == "__main__":
    raise NotImplementedError
