import pandas as pd
from matplotlib import pyplot as plt

from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.detect_echoes import get_analytic_signal, get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import (
    INTERPOLATION_FACTOR,
    ORIGINAL_SAMPLE_RATE,
    SAMPLE_RATE,
)


def generate_signals_for_matlab(
    measurements: pd.DataFrame = None,
    center_frequency: float = 25000,
    t_var: float = 0.0001,
    propagation_speed_mps: float = 1000,
    crop_end: float = 0.000637,
    number_of_sensors: int = 10,
):
    if measurements is None:
        FILE_NAME, measurements = import_the_data()
        measurements = do_measurement_preprocessing(measurements)
    else:
        FILE_NAME = f"generated_signal_{center_frequency // 1000}kHz_{t_var}s_{propagation_speed_mps}mps_{number_of_sensors}sensors_interp{INTERPOLATION_FACTOR}"
        if crop_end is not None:
            measurements = crop_data(
                measurements,
                time_start=2.5002,
                time_end=2.5002 + crop_end,
            )
    measurements = drop_actuator_channel(measurements)
    plot_them_time_signals(measurements)
    analytic_signals = get_analytic_signal(measurements)
    # Ask user if they want to save the analytic signals
    if input("Save analytic signals? y/n: ") == "y":
        make_a_nice_csv_file(FILE_NAME, analytic_signals)
    envelopes = get_envelopes(measurements)
    plot_them_envelopes(envelopes)


def drop_actuator_channel(
    measurements,
):
    return measurements.drop(columns=["Actuator"])


def import_the_data():
    FILE_FOLDER = "Plate_10mm/Setup4/touch"
    FILE_NAME = "nik_touch_35_35_v1"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)
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
    fig, axs = plt.subplots(1, 2, squeeze=False)
    for sensor in measurements.columns:
        compare_signals(
            fig,
            axs,
            [measurements[sensor]],
            plots_to_plot=["time", "fft"],
            sharey=True,
        )
    fig.suptitle("Signals before analytic signal processing")


def plot_them_envelopes(
    envelopes,
):
    fig, axs = plt.subplots(1, 1, squeeze=False)
    for sensor in envelopes.columns:
        compare_signals(
            fig,
            axs,
            [envelopes[sensor]],
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
    # Save the analytic signals in a csv file
    analytic_signals.to_csv(
        f"{FILE_NAME}.csv",
        index=False,
        header=False,
    )

    """Save the analytic signals in a csv file
    without parenthesis around the complex numbers.
    """
    with open(f"{FILE_NAME}.csv", "r") as f:
        lines = f.readlines()
    with open(f"{FILE_NAME}.csv", "w") as f:
        for line in lines:
            f.write(line.replace("(", "").replace(")", ""))
