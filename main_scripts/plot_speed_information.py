from matplotlib import pyplot as plt
import pandas as pd
from utils.csv_to_df import make_dataframe_from_csv
from utils.global_constants import FIGURES_SAVE_PATH


def plot_speed_information():
    # Import csv file with speed information as dataframes
    velocities_A0_df = pd.read_csv("simulated_speeds_for_A0.csv")
    velocities_S0_df = pd.read_csv("simulated_speeds_for_S0.csv")

    fig, ax = plt.subplots()

    velocities_A0_df.plot(
        ax=ax,
        x="A0 f (kHz)",
        y=["A0 Phase velocity (m/ms)", "A0 Energy velocity (m/ms)"],
        xlabel="Frequency (kHz)",
        ylabel="Phase velocity (m/ms)",
        legend=True,
        grid=True,
        xlim=(0, 60),
        style={
            "A0 Energy velocity (m/ms)": "--",
            "A0 Phase velocity (m/ms)": "C0",
        },
    )

    velocities_S0_df.plot(
        ax=ax,
        x="S0 f (kHz)",
        y=["S0 Phase velocity (m/ms)", "S0 Energy velocity (m/ms)"],
        xlabel="Frequency (kHz)",
        ylabel="Phase velocity (m/ms)",
        legend=True,
        grid=True,
        xlim=(0, 60),
        style={
            "S0 Energy velocity (m/ms)": "--",
            "S0 Phase velocity (m/ms)": "C1",
        },
    )
    # Save figure as pdf file in the path specified by FIGURES_SAVE_PATH
    fig.savefig(f"{FIGURES_SAVE_PATH}/phase_and_energy_speeds_for_A0_and_S0.pdf")

    fig, ax = plt.subplots()

    velocities_A0_df.plot(
        ax=ax,
        x="A0 f (kHz)",
        y="A0 Wavelength (mm)",
        xlabel="Frequency (kHz)",
        ylabel="Wavelength (mm)",
        legend=True,
        grid=True,
        xlim=(0, 60),
        ylim=(0, 50),
    )

    velocities_S0_df.plot(
        ax=ax,
        x="S0 f (kHz)",
        y="S0 Wavelength (mm)",
        xlabel="Frequency (kHz)",
        ylabel="Wavelength (mm)",
        legend=True,
        grid=True,
        xlim=(0, 60),
        ylim=(0, 50),
    )
    fig.savefig(f"{FIGURES_SAVE_PATH}/wavelengths_for_A0_and_S0.pdf")

    return 0
