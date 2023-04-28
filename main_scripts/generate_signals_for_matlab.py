import numpy as np
from matplotlib import pyplot as plt

from utils.csv_to_df import make_dataframe_from_csv
from utils.data_processing.detect_echoes import get_analytic_signal, get_envelopes
from utils.data_processing.preprocessing import crop_data, crop_to_signal, filter_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import compare_signals
from utils.global_constants import ORIGINAL_SAMPLE_RATE
from utils.plate_setups import Setup4


def generate_signals_for_matlab():
    SETUP = Setup4(actuator_coordinates=np.array([0.35, 0.35]))
    FILE_NAME, measurements = import_the_data()
    measurements = do_the_preprocessing(measurements)
    plot_them_time_signals(measurements)

    measurements = drop_actuator_channel(measurements)
    analytic_signals = get_analytic_signal(measurements)
    # Save the analytic signals in a csv file without parenthesis around the complex numbers
    make_a_nice_csv_file(FILE_NAME, analytic_signals)

    envelopes = get_envelopes(measurements)
    plot_them_envelopes(envelopes)


def drop_actuator_channel(measurements):
    return measurements.drop(columns=["Actuator"])


def import_the_data():
    FILE_FOLDER = "Plate_10mm/Setup4/touch"
    FILE_NAME = "nik_touch_35_35_v1"
    measurements = make_dataframe_from_csv(file_folder=FILE_FOLDER, file_name=FILE_NAME)
    return FILE_NAME, measurements


def do_the_preprocessing(measurements):
    measurements["Actuator"] = 0
    measurements = crop_to_signal(measurements)
    CRITICAL_FREQUENCY = 35000
    measurements = filter_signal(
        measurements,
        filtertype="bandpass",
        critical_frequency=CRITICAL_FREQUENCY,
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


def plot_them_time_signals(measurements):
    fig, axs = plt.subplots(1, 2, squeeze=False)
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 1"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 2"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [measurements["Sensor 3"]],
        plots_to_plot=["time", "fft"],
        sharey=True,
    )
    axs[0, 0].legend(["Sensor 1", "Sensor 2", "Sensor 3"], loc="upper right")


def plot_them_envelopes(envelopes):
    fig, axs = plt.subplots(1, 1, squeeze=False)
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 1"]],
        plots_to_plot=["time"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 2"]],
        plots_to_plot=["time"],
        sharey=True,
    )
    compare_signals(
        fig,
        axs,
        [envelopes["Sensor 3"]],
        plots_to_plot=["time"],
        sharey=True,
    )


def make_a_nice_csv_file(FILE_NAME, analytic_signals):
    # Make a full file name that is "FILE_NAME + today's date and time as HH-MM-SS"
    TODAYS_DATE = pd.Timestamp.now().strftime("%Y_%m_%d")
    TODAYS_TIME = pd.Timestamp.now().strftime("%H_%M_%S")
    FILE_NAME = f"{FILE_NAME}_{TODAYS_DATE}_{TODAYS_TIME}_analytic"
    # Save the analytic signals in a csv file
    analytic_signals.to_csv(f"{FILE_NAME}.csv")

    """Save the analytic signals in a csv file
    without parenthesis around the complex numbers.
    """
    with open(f"{FILE_NAME}.csv", "r") as f:
        lines = f.readlines()
    with open(f"{FILE_NAME}.csv", "w") as f:
        for line in lines:
            f.write(line.replace("(", "").replace(")", ""))
