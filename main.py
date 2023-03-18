"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import matplotlib.pyplot as plt

from main_scripts.inspect_touch_release import inspect_touch_release
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()
    inspect_touch_release()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
