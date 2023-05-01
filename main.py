"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


from matplotlib import pyplot as plt
from main_scripts.estimate_touch_location import estimate_touch_location
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    # Call one of the functions found in /main_scripts
    estimate_touch_location(
        center_frequency=25000,
        t_var=4e-9,
        propagation_speed_mps=992,
        snr_dB=10,
        attenuation_dBpm=15,
    )

    # Plot all figures at the same time
    plt.show()


if __name__ == "__main__":
    main()
