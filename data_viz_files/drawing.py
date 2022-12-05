import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict

from data_processing.detect_echoes import (find_mirrored_source,
                                           flip_sensors,
                                           flip_sources)
from objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor


def plot_legend_without_duplicates(placement: str = None):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    if placement:
        plt.legend(by_label.values(), by_label.keys(), loc=placement)
    else:
        plt.legend(by_label.values(), by_label.keys())


def plot_legend_without_duplicates_ax(ax):
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')


def draw_setup_ideal():
    table = Table()

    sensor_2 = Sensor(coordinates=np.array([table.LENGTH / 2,
                                            table.WIDTH - 0.05]))
    sensor_1 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y))
    sensor_3 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y))
    sensor_4 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y - 0.02))
    sensor_5 = Sensor(coordinates=(sensor_2.x, sensor_2.y - 0.02))
    sensor_6 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y - 0.02))
    sensor_7 = Sensor(coordinates=(sensor_2.x - 0.02, sensor_2.y - 0.04))
    sensor_8 = Sensor(coordinates=(sensor_2.x, sensor_2.y - 0.04))
    sensor_9 = Sensor(coordinates=(sensor_2.x + 0.02, sensor_2.y - 0.04))

    actuator = Actuator(coordinates=np.array([sensor_5.x - 0.10,
                                              sensor_5.y]))

    plt.axes()
    table.draw()
    actuator.draw()
    sensor_2.draw()
    sensor_1.draw()
    sensor_3.draw()
    sensor_4.draw()
    sensor_5.draw()
    sensor_6.draw()
    sensor_7.draw()
    sensor_8.draw()
    sensor_9.draw()

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()

    return actuator, sensor_1, sensor_2, sensor_3


def draw_a_setup(sources: np.ndarray, sensors: np.ndarray):
    table = Table()

    EDGES_TO_FLIP_AROUND = np.array([table.BOTTOM_EDGE,
                                     table.LEFT_EDGE])

    mirrored_sources = np.array([])
    mirrored_sources = np.append(mirrored_sources,
                                 find_mirrored_source(sources[0], [edge for edge in EDGES_TO_FLIP_AROUND]))

    plt.axes()
    table.draw()
    # sources[0].draw()
    [mirrored_source.draw() for mirrored_source in mirrored_sources]
    sources = flip_sources(sources, EDGES_TO_FLIP_AROUND)
    sensors = flip_sensors(sensors, EDGES_TO_FLIP_AROUND)
    [source.draw() for source in sources]
    [sensor.draw() for sensor in sensors[0:1]]
    line_mirr_source = plt.Line2D((mirrored_sources[0].x, sensors[0].x),
                                  (mirrored_sources[0].y, sensors[0].y),
                                  color='black',
                                  lw=0.75,
                                  linestyle='--',
                                  zorder=2)
    plt.gca().add_patch(line_mirr_source)


    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # plt.title(f'Visualising a mirrored source\nfor bouncing on edges: {EDGES_TO_FLIP_AROUND}')
    plt.title(f'Visualising a mirrored source\nfor bouncing on edges: bottom and left side')
    plot_legend_without_duplicates()

    plt.show()

    return
