import numpy as np
import matplotlib.pyplot as plt


def draw_table():
    TABLE_LENGTH = 0.716    # m
    TABLE_WIDTH = 0.597     # m
    TABLE_SURFACE_COLOUR = '#fbe5b6'
    TABLE_LINE_COLOUR = '#f0c18b'

    table = plt.Rectangle((0, 0),
                          TABLE_LENGTH,
                          TABLE_WIDTH,
                          fc=TABLE_SURFACE_COLOUR,
                          ec=TABLE_LINE_COLOUR,
                          lw=2,
                          zorder=0)

    plt.gca().add_patch(table)

    for i in range(1, 3):
        line_x = plt.Line2D((i * TABLE_LENGTH / 3, i * TABLE_LENGTH / 3),
                            (0, TABLE_WIDTH),
                            color=TABLE_LINE_COLOUR,
                            lw=0.75,
                            linestyle='--',
                            zorder=1)
        line_y = plt.Line2D((0, TABLE_LENGTH),
                            (i * TABLE_WIDTH / 3, i * TABLE_WIDTH / 3),
                            color=TABLE_LINE_COLOUR,
                            lw=0.75,
                            linestyle='--',
                            zorder=1)
        plt.gca().add_patch(line_x)
        plt.gca().add_patch(line_y)


def draw_actuator(actuator_coord):
    ACTUATOR_COLOUR = '#D4434A'
    actuator = plt.Circle(actuator_coord,
                          radius=0.01,
                          fc=ACTUATOR_COLOUR,
                          ec='dimgray')
    plt.gca().add_patch(actuator)


def draw_sensor(actuator_coord):
    ACTUATOR_COLOUR = '#AEAFA7'
    actuator = plt.Circle(actuator_coord,
                          radius=0.01,
                          fc=ACTUATOR_COLOUR,
                          ec='dimgray')
    plt.gca().add_patch(actuator)