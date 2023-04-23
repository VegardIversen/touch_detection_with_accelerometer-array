from matplotlib import pyplot as plt
from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.preprocessing import (
    crop_data,
    crop_to_signal,
    filter_signal,
)
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import SAMPLE_RATE
from utils.plate_setups import Setup1


def find_propagation_velocities():
    measurements = import_the_file()
    measurements = crop_to_signal(measurements)
    SETUP = Setup1()
    SETUP.draw()
    measurements = interpolate_signal(measurements)
    find_speed_for_15kHz(measurements, SETUP)
    find_speed_for_35kHz(measurements, SETUP)


def find_speed_for_15kHz(measurements, SETUP):
    # Keeping frequencies in separate functions as they need different cropping
    CRITICAL_FREQUENCY = 15000
    filtered_measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
        plot_response=False,
        order=2,
        sample_rate=SAMPLE_RATE,
    )
    # filtered_measurements = crop_to_signal(filtered_measurements)
    filtered_measurements = crop_data(
        filtered_measurements,
        time_start=0.04821,
        time_end=0.04821 + 0.0009,
        sample_rate=SAMPLE_RATE,
    )
    propagation_speed = SETUP.get_propagation_speed(filtered_measurements)
    print(f"Propagation speed: {propagation_speed:.2f} m/s at {CRITICAL_FREQUENCY} Hz")

    fig, axs = plt.subplots(3, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [
            filtered_measurements[sensor]
            for sensor in ["Sensor 3", "Sensor 2", "Sensor 1"]
        ],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )

    fig, axs = plt.subplots(1, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 1"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 2"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 3"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )


def find_speed_for_35kHz(measurements, SETUP):
    # Keeping frequencies in separate functions as they need different cropping
    CRITICAL_FREQUENCY = 35000
    filtered_measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
        plot_response=False,
        order=2,
        sample_rate=SAMPLE_RATE,
    )
    # filtered_measurements = crop_to_signal(filtered_measurements)
    filtered_measurements = crop_data(
        filtered_measurements,
        time_start=0.04821,
        time_end=0.04821 + 0.000587,
        sample_rate=SAMPLE_RATE,
    )
    propagation_speed = SETUP.get_propagation_speed(filtered_measurements)
    print(f"Propagation speed: {propagation_speed:.2f} m/s at {CRITICAL_FREQUENCY} Hz")

    fig, axs = plt.subplots(3, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [
            filtered_measurements[sensor]
            for sensor in ["Sensor 3", "Sensor 2", "Sensor 1"]
        ],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )

    fig, axs = plt.subplots(1, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 1"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 2"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 3"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )


def import_the_file():
    file_folder = "Plate_10mm/Setup1/touch"
    file_name = "nik_touch_v2"
    measurements = make_dataframe_from_csv(file_folder=file_folder, file_name=file_name)
    measurements["Actuator"] = 0
    return measurements
