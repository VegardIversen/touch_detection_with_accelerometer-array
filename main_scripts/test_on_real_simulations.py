from os import path

import numpy as np
import pandas as pd


def test_on_real_simulations():
    simulation_data = import_simulation_data()


def import_simulation_data():
    # Set the path to your data file
    data_file = "az_on_plate_top_Teflon_25kHz_pulse_5cmfromedge.txt"

    # Read the file into a Pandas DataFrame
    simulation_data = pd.read_csv(
        data_file,
        delim_whitespace=True,
        comment="%",
    )

    # Drop x and z columns
    simulation_data.drop(
        columns=[simulation_data.columns[0], simulation_data.columns[2]],
        inplace=True,
    )

    # Transpose dataframe
    simulation_data = simulation_data.T

    # Save as a csv file named "simulation_data_formatted.csv"
    simulation_data.to_csv(
        "simulation_data_formatted.csv",
        header=False,
        index=False,
    )

    return simulation_data
