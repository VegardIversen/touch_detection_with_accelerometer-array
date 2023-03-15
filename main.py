"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt

from main_scripts.inspect_10mm_plate_touch import inspect_touch
from main_scripts.linear_array_by_edge import linear_array_by_edge
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()
    inspect_touch()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
