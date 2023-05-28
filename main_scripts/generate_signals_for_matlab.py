import pandas as pd
from matplotlib import pyplot as plt
import scipy

from utils.csv_to_df import import_measurements
from utils.data_processing.detect_echoes import get_analytic_signal, get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import (
    INTERPOLATION_FACTOR,
    ORIGINAL_SAMPLE_RATE,
    SAMPLE_RATE,
)
import os


def generate_signals_for_matlab(
    parameters: dict,
    measurements: pd.DataFrame = None,
    center_frequency_Hz: float = 25000,
    t_var: float = 0.0001,
    propagation_speed_mps: float = 1000,
    crop_end_s: float = 0.000637,
    number_of_sensors: int = 10,
    array_type: str = "ula",
):
    FILE_NAME = f"comsol_simulations_analytic_signals_{array_type}"
    if measurements is None:
        FILE_NAME = f"niklas_simulations/generated_signal_{center_frequency_Hz // 1000}kHz_{t_var}s_{propagation_speed_mps}mps_{number_of_sensors}sensors_interp{INTERPOLATION_FACTOR}"
        if crop_end_s is not None:
            measurements = crop_data(
                measurements,
                time_start=2.5002,
                time_end=2.5002 + crop_end_s,
            )
    measurements = drop_actuator_channel(measurements)
    plot_them_time_signals(measurements)
    analytic_signals = get_analytic_signal(measurements)
    envelopes = get_envelopes(measurements)
    plot_them_envelopes(envelopes)

    # Ask user if they want to save the analytic signals
    if input("Save analytic signals? y/n: ") == "y":
        make_a_nice_csv_file(
            FILE_NAME,
            analytic_signals,
        )
        make_a_nice_parameters_file(parameters)
        print(f"\nSaved analytic signals to {FILE_NAME}.csv\n")
        exit()
    return analytic_signals


def drop_actuator_channel(
    measurements,
):
    # If there is a channel for the actuator, drop it
    if "Actuator" in measurements.columns:
        measurements = measurements.drop(columns=["Actuator"])
    return measurements


def import_the_data():
    # TODO: Do this in main()
    FILE_FOLDER = "Plate_10mm/Setup4/touch"
    FILE_NAME = "nik_touch_35_35_v1"
    measurements = import_measurements(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    return FILE_NAME, measurements


def do_measurement_preprocessing(
    measurements,
    frequency,
):
    measurements["Actuator"] = 0
    measurements = crop_to_signal(measurements)
    measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=frequency,
        plot_response=False,
        order=2,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    measurements = crop_to_signal(measurements)
    measurements = crop_data(
        measurements,
        time_start=0.00265,
        time_end=0.00289,
        sample_rate=ORIGINAL_SAMPLE_RATE,
    )
    measurements = interpolate_signal(measurements)
    return measurements


def plot_them_time_signals(
    measurements,
):
    fig, axs = plt.subplots(
        nrows=measurements.shape[1],
        ncols=2,
        squeeze=False,
    )
    compare_signals(
        fig,
        axs,
        [measurements[channel] for channel in measurements.columns],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    fig.suptitle("Signals before analytic signal processing")


def plot_them_envelopes(
    envelopes,
):
    fig, axs = plt.subplots(
        nrows=envelopes.shape[1],
        ncols=1,
        squeeze=False,
    )
    compare_signals(
        fig,
        axs,
        [envelopes[channel] for channel in envelopes.columns],
        plots_to_plot=["time"],
        sharey=True,
    )
    fig.suptitle("Envelopes of exported analytic signals")


def make_a_nice_csv_file(
    FILE_NAME,
    analytic_signals,
    add_timestamp=False,
):
    if add_timestamp:
        # Make a full file name that is "FILE_NAME + today's date and time as HH-MM-SS"
        TODAYS_DATE = pd.Timestamp.now().strftime("%Y_%m_%d")
        TODAYS_TIME = pd.Timestamp.now().strftime("%H_%M_%S")
        FILE_NAME = f"{FILE_NAME}_{TODAYS_DATE}_{TODAYS_TIME}_analytic"
    FOLDER_NAME = "matlab"
    # Save the analytic signals in a csv file
    analytic_signals.to_csv(
        f"{FOLDER_NAME}/{FILE_NAME}.csv",
        index=False,
        header=False,
    )

    """Save the analytic signals in a csv file
    without parenthesis around the complex numbers.
    """
    with open(f"{FOLDER_NAME}/{FILE_NAME}.csv", "r") as f:
        lines = f.readlines()
    with open(f"{FOLDER_NAME}/{FILE_NAME}.csv", "w") as f:
        for line in lines:
            f.write(line.replace("(", "").replace(")", ""))

    # Check if path exists
    if os.path.exists(
        "/mnt/c/Users/nikla/Documents/GitHub/touch_detection_with_accelerometer-array/matlab/"
    ):
        os.system(
            f"cp {FOLDER_NAME}/{FILE_NAME}.csv"
            "/mnt/c/Users/nikla/Documents/GitHub/touch_detection_with_accelerometer-array/matlab/"
        )


def make_a_nice_parameters_file(parameters):
    scipy.io.savemat(
        "matlab/parameters.mat",
        parameters,
    )
    if os.path.exists(
        "/mnt/c/Users/nikla/Documents/GitHub/touch_detection_with_accelerometer-array/matlab/"
    ):
        os.system(
            "cp matlab/parameters.mat"
            "/mnt/c/Users/nikla/Documents/GitHub/touch_detection_with_accelerometer-array/matlab/"
        )
