from matplotlib import pyplot as plt
import pandas as pd
from utils.csv_to_df import make_dataframe_from_csv
from utils.global_constants import FIGURES_SAVE_PATH


def plot_speed_information(tonni_params: bool = True):
    # Import csv file with speed information as dataframes
    VELOCITIES_A0_DF = pd.read_csv("simulated_speeds_for_A0_with_Tonni_params.csv")
    VELOCITIES_S0_DF = pd.read_csv("simulated_speeds_for_S0_with_Tonni_params.csv")

    fig, ax = plt.subplots()

    VELOCITIES_A0_DF.plot(
        ax=ax,
        x="A0 f (kHz)",
        y=["A0 Phase velocity (m/ms)", "A0 Energy velocity (m/ms)"],
        xlabel="Frequency (kHz)",
        ylabel="Velocity (m/ms)",
        xlim=(0, 100),
        ylim=(0, 1.2),
        style={
            "A0 Energy velocity (m/ms)": "--",
            "A0 Phase velocity (m/ms)": "C0",
        },
    )

    VELOCITIES_S0_DF.plot(
        ax=ax,
        x="S0 f (kHz)",
        y=["S0 Phase velocity (m/ms)", "S0 Energy velocity (m/ms)"],
        xlabel="Frequency (kHz)",
        grid=True,
        style={
            "S0 Energy velocity (m/ms)": "--",
            "S0 Phase velocity (m/ms)": "C1",
        },
    )

    ax.legend(
        [
            "A0 Phase velocity",
            "A0 Energy velocity",
            "S0 Phase velocity",
            "S0 Energy velocity",
        ],
        # loc="upper right",
    )

    # Save figure as pdf file in the path specified by FIGURES_SAVE_PATH
    fig.savefig(
        f"{FIGURES_SAVE_PATH}/phase_and_energy_speeds_for_A0_and_S0.pdf",
        bbox_inches="tight",
    )

    fig, ax = plt.subplots()

    VELOCITIES_A0_DF.plot(
        ax=ax,
        x="A0 f (kHz)",
        y="A0 Wavelength (mm)",
        xlabel="Frequency (kHz)",
        ylabel="Wavelength (mm)",
        # title=PLOT_TITLE,
        xlim=(0, 100),
        ylim=(0, 50),
    )

    VELOCITIES_S0_DF.plot(
        ax=ax,
        x="S0 f (kHz)",
        y="S0 Wavelength (mm)",
        xlabel="Frequency (kHz)",
        grid=True,
    )

    ax.legend(
        [
            "A0 Wavelength",
            "S0 Wavelength",
        ]
    )

    fig.savefig(
        f"{FIGURES_SAVE_PATH}/wavelengths_for_A0_and_S0.pdf",
        bbox_inches="tight",
    )

    return 0
