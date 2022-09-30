from data_processing.cross_correlation_position import FindTouchPosition
from data_viz_files.display_table_grid import draw
from pathlib import Path

print(__file__)

DATA_FOLDER = f'{Path.home()}\\OneDrive - NTNU\\NTNU\\ProsjektOppgave'

if __name__ == '__main__':
    cell = FindTouchPosition(DATA_FOLDER + '\\first_test_touch_passive_setup2\\touch_test_passive_setup2_place_B3_center_v2.csv')
    draw(cell)
