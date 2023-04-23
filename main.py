"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt
from main_scripts.generate_signals_for_matlab import generate_signals_for_matlab

from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    generate_signals_for_matlab()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
