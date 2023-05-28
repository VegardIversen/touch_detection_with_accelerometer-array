from matplotlib import pyplot as plt

from utils.csv_to_df import import_measurements
from utils.data_processing.detect_echoes import get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
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
    find_speed_for_25kHz(measurements, SETUP)
    # find_speed_for_15kHz(measurements, SETUP)
    # find_speed_for_35kHz(measurements, SETUP)


def find_speed_for_5kHz(measurements, SETUP):
    # Keeping frequencies in separate functions as they need different cropping
    CRITICAL_FREQUENCY = 5000
    filtered_measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
        plot_response=False,
        order=2,
        sample_rate=SAMPLE_RATE,
        q=0.05,
    )
    # filtered_measurements = crop_to_signal(filtered_measurements)
    TIME_START = 0.0
    TIME_END = 0.1
    filtered_measurements = crop_data(
        filtered_measurements,
        time_start=TIME_START,
        time_end=TIME_END,
        sample_rate=SAMPLE_RATE,
    )
    propagation_speed = SETUP.get_propagation_speed(filtered_measurements)
    print(f"Propagation speed: {propagation_speed:.2f} m/s at {CRITICAL_FREQUENCY} Hz")

    envelopes = get_envelopes(filtered_measurements)
    # Find the five indices of the highest values in the envelopes
    max_indices = {
        sensor: envelopes[sensor].nlargest(5).index.values
        for sensor in ["Sensor 1", "Sensor 2", "Sensor 3"]
    }
    max_times = {
        sensor: max_indices[sensor] / SAMPLE_RATE
        for sensor in ["Sensor 1", "Sensor 2", "Sensor 3"]
    }
    print(f"Times of maximum values: {max_times}")
    sensor_2_1_diff = max_times["Sensor 2"][1] - max_times["Sensor 1"][0]
    sensor_3_2_diff = max_times["Sensor 3"][0] - max_times["Sensor 2"][1]
    sensor_3_1_diff = max_times["Sensor 3"][0] - max_times["Sensor 1"][0]
    print(f"Time difference between Sensor 2 and 1: {sensor_2_1_diff:.6f} s")
    print(f"Time difference between Sensor 3 and 2: {sensor_3_2_diff:.6f} s")
    print(
        f"Time difference between Sensor 3 and 1 (as 0.1 m): {(sensor_3_1_diff / 2):.6f} s"
    )
    print(f"Propagation speed between Sensor 2 and 1: {0.1 / sensor_2_1_diff} m/s")
    print(f"Propagation speed between Sensor 3 and 2: {0.1 / sensor_3_2_diff} m/s")
    print(f"Propagation speed between Sensor 3 and 1: {0.2 / sensor_3_1_diff} m/s")

    fig, axs = plt.subplots(3, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [envelopes[sensor] for sensor in ["Sensor 1", "Sensor 2", "Sensor 3"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )

    fig, axs = plt.subplots(1, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [filtered_measurements["Sensor 3"]],
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
        [filtered_measurements["Sensor 1"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )


def find_speed_for_10kHz(measurements, SETUP):
    # Keeping frequencies in separate functions as they need different cropping
    CRITICAL_FREQUENCY = 10000
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
        time_end=0.04821 + 0.01,
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
    TIME_START = 0.04821
    TIME_END = 5
    filtered_measurements = crop_data(
        filtered_measurements,
        time_start=TIME_START,
        time_end=TIME_END,
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


def find_speed_for_25kHz(measurements, SETUP):
    # Keeping frequencies in separate functions as they need different cropping
    CRITICAL_FREQUENCY = 25000
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
        time_start=0.00 + 0,
        time_end=0.04825 + 0.001,
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
    TIME_START = 0.04821
    TIME_END = 5
    filtered_measurements = crop_data(
        filtered_measurements,
        time_start=TIME_START,
        time_end=TIME_END,
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
    measurements = import_measurements(file_folder=file_folder, file_name=file_name)
    measurements["Actuator"] = 0
    return measurements
