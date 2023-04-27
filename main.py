"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
from main_scripts.find_propagation_velocities import import_the_file
from main_scripts.generate_ideal_signal import compare_to_ideal_signal
from utils.data_processing.preprocessing import crop_to_signal
from utils.data_processing.processing import interpolate_signal
from utils.data_visualization.visualize_data import set_fontsizes
from utils.plate_setups import Setup1


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    measurements = import_the_file()
    measurements = crop_to_signal(measurements)
    SETUP = Setup1()
    SETUP.draw()
    measurements = interpolate_signal(measurements)
    interpolate_signal(measurements)
    compare_to_ideal_signal(
        setup=SETUP,
        measurements=measurements,
        filtertype="bandpass",
        critical_frequency=20000,
        attenuation_dBpm=15,
        propagation_speed_mps=1050,
        signal_model="gaussian",
    )
    plt.show()
    return 0


if __name__ == "__main__":
    main()
