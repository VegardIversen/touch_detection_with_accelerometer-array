"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


import matplotlib
from matplotlib import pyplot as plt

from main_scripts.inspect_10mm_plate_touch import inspect_swipe, inspect_touch
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()

    matplotlib.use("TkAgg")
    # * Call one of the functions found in /main_scripts

    inspect_touch()
    plt.show()


if __name__ == "__main__":
    main()
