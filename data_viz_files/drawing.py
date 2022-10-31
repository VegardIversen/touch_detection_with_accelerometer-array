from unicodedata import mirrored
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict

from data_processing.detect_echoes import find_mirrored_source, flip_sensors, flip_sources
from objects import MirroredSensor, MirroredSource, Table, Actuator, Sensor


def draw_table(plot_lines=True):
    "Draw the table with the real dimensions, including the lines."
    table = patches.Rectangle((0, 0),
                              Table.LENGTH,
                              Table.WIDTH,
                              fc=Table.SURFACE_COLOUR,
                              ec=Table.LINE_COLOUR,
                              lw=2,
                              zorder=0)

    plt.gca().add_patch(table)

    if plot_lines:
        for i in range(1, 3):
            line_x = plt.Line2D((i / 3 * Table.LENGTH, i / 3 * Table.LENGTH),
                                (0, Table.WIDTH),
                                color=Table.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            line_y = plt.Line2D((0, Table.LENGTH),
                                (i / 3 * Table.WIDTH, i / 3 * Table.WIDTH),
                                color=Table.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


def draw_actuator(actuator_coord: np.array):

    actuator = plt.Circle(actuator_coord,
                          radius=Actuator.RADIUS,
                          fc=Actuator.FILL_COLOUR,
                          ec=Actuator.EDGE_COLOUR,
                          label='Actuator/mirrored source')
    plt.gca().add_patch(actuator)


def draw_sensor(actuator_coord):
    SENSOR_COLOUR = '#AEAFA7'
    sensor = plt.Circle(actuator_coord,
                        radius=0.01,
                        fc=SENSOR_COLOUR,
                        ec='dimgray',
                        label='Sensor')
    plt.gca().add_patch(sensor)


def draw_waves(actuator_coord: np.array, n_waves=3):
    """Draw arcs representing the waves.
    TODO:   Automate to have the correct wavelength (and amplitude?)
            for a given frequency. Might also be better with circles.
    """
    ARC_COLOUR = '#f0c18b'
    for i in range(1, n_waves + 1):
        arc = patches.Arc(actuator_coord,
                          0.07 * i,     # could be automated with wavelengths
                          0.07 * i,     # could be automated with wavelengths
                          theta1=0,
                          theta2=180,
                          color=ARC_COLOUR,
                          lw=0.75,
                          linestyle='--')
        plt.gca().add_patch(arc)


def plot_legend_without_duplicates():
    """Avoid duplicate labels in the legend"""
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def draw_arrow(start_coord, end_coord, subtract_length=0):
    """Draw an arrow between two points.
    The arrow is drawn from the start_coord to the end_coord.
    """
    ARROW_SENSOR_OFFSET = end_coord - start_coord
    plt.arrow(start_coord[0], start_coord[1],
              ARROW_SENSOR_OFFSET[0], ARROW_SENSOR_OFFSET[1],
              color='black',
              head_width=0.03,
              length_includes_head=True)


def draw_setup_1():
    table = Table()

    actuator = Actuator(coordinates=np.array([1 / 2 * table.LENGTH,
                                              1 / 6 * table.WIDTH]))

    sensor_2 = Sensor(coordinates=np.array([table.LENGTH / 2,
                                            table.WIDTH - 0.05]))
    SENSOR_1_OFFSET = np.array([-0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    SENSOR_3_OFFSET = np.array([0.08 / 2, -(np.sqrt(0.08 ** 2 - 0.04 ** 2))])
    sensor_1 = Sensor(coordinates=(sensor_2.coordinates + SENSOR_1_OFFSET))
    sensor_3 = Sensor(coordinates=(sensor_2.coordinates + SENSOR_3_OFFSET))

    plt.axes()
    table.draw()
    actuator.draw()
    sensor_2.draw()
    sensor_1.draw()
    sensor_3.draw()

    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plot_legend_without_duplicates()

    plt.show()


def draw_a_setup():
    table = Table()

    actuator = Actuator(coordinates=np.array([1 / 6 * table.LENGTH,
                                              1 / 2 * table.WIDTH]))
    sensor_1 = Sensor(coordinates=np.array([2 / 3 * table.LENGTH,
                                            5 / 6 * table.WIDTH]))
    sensor_2 = Sensor(coordinates=np.array([2 / 3 * table.LENGTH,
                                            1 / 3 * table.WIDTH]))

    sources = np.array([actuator])
    sensors = np.array([sensor_1])

    EDGES_TO_FLIP_AROUND = np.array([table.LEFT_EDGE,
                                     table.TOP_EDGE])

    mirrored_sources = np.array([])
    mirrored_sources = np.append(mirrored_sources, find_mirrored_source(actuator, [edge for edge in EDGES_TO_FLIP_AROUND]))

    plt.axes()
    table.draw()
    actuator.draw()
    [mirrored_source.draw() for mirrored_source in mirrored_sources]
    # sources = flip_sources(sources, EDGES_TO_FLIP_AROUND)
    # sensors = flip_sensors(sensors, EDGES_TO_FLIP_AROUND)
    [source.draw() for source in sources]
    [sensor.draw() for sensor in sensors]
    line_mirr_source = plt.Line2D((mirrored_sources[0].x, sensor_1.x),
                                  (mirrored_sources[0].y, sensor_1.y),
                                  color='black',
                                  lw=0.75,
                                  linestyle='--',
                                  zorder=2)
    plt.gca().add_patch(line_mirr_source)


    plt.axis('scaled')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Visualising a mirrored source\nfor bouncing on edges: {EDGES_TO_FLIP_AROUND}')
    plot_legend_without_duplicates()

    plt.show()

    return
