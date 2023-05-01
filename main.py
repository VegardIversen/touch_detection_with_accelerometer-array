"""Author: Niklas Strømsnes
Date: 2022-01-09
"""


from main_scripts.estimate_touch_location import estimate_touch_location
from utils.data_visualization.visualize_data import set_fontsizes


def main():
    set_fontsizes()
    # Call one of the functions found in /main_scripts
    estimate_touch_location()


if __name__ == "__main__":
    main()
