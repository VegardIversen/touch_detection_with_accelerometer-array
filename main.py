"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import matplotlib
from matplotlib import pyplot as plt

from main_scripts.full_touch_localization import full_touch_localization
from main_scripts.test_on_real_simulations import prepare_simulation_data
from main_scripts.inspect_10mm_plate_touch import inspect_swipe, inspect_touch
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    # matplotlib.use("TkAgg")
    # * Call one of the functions found in /main_scripts

    full_touch_localization()
    plt.show()


if __name__ == "__main__":
    main()
