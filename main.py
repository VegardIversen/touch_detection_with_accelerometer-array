"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
import numpy as np
from main_scripts.generate_ideal_signal import generate_ideal_signal
from utils.data_visualization.visualize_data import set_fontsizes
from utils.plate_setups import Setup5


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    SETUP = Setup5(
        actuator_coordinates=[0.5, 0.35],
        array_start_coordinates=np.array([0.05, 0.65]),
    )
    SETUP.draw()
    generate_ideal_signal(
        setup=SETUP,
        attenuation_dBpm=10,
        propagation_speed_mps=600,
        cutoff_frequency=35000,
        signal_length_s=0.125,
        signal_model="line",
    )
    plt.show()
    return 0


if __name__ == "__main__":
    main()
