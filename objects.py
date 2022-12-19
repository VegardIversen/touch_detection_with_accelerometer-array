"""Objects such as the table, acutators and sensors have their own class."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Table:
    """Represents the table and its edges."""
    """Table dimensions"""
    LENGTH = 0.80   # m
    WIDTH = 0.60    # m
    """Enum for representing edges"""
    TOP_EDGE = 1
    RIGHT_EDGE = 2
    BOTTOM_EDGE = 3
    LEFT_EDGE = 4
    """Colour settings for drawing"""
    SURFACE_COLOUR = '#fbe5b6'
    LINE_COLOUR = '#f0c18b'

    """Table locations, in the middle of each block"""
    A1 = np.array([1 / 6 * LENGTH, 1 / 6 * WIDTH])
    A2 = np.array([3 / 6 * LENGTH, 1 / 6 * WIDTH])
    A3 = np.array([5 / 6 * LENGTH, 1 / 6 * WIDTH])
    B1 = np.array([1 / 6 * LENGTH, 3 / 6 * WIDTH])
    B2 = np.array([3 / 6 * LENGTH, 3 / 6 * WIDTH])
    B3 = np.array([5 / 6 * LENGTH, 3 / 6 * WIDTH])
    C1 = np.array([1 / 6 * LENGTH, 5 / 6 * WIDTH])
    C2 = np.array([3 / 6 * LENGTH, 5 / 6 * WIDTH])
    C3 = np.array([5 / 6 * LENGTH, 5 / 6 * WIDTH])

    def draw(self):
        "Draw the table with the real dimensions, including the lines."
        table = patches.Rectangle((0, 0),
                                  self.LENGTH,
                                  self.WIDTH,
                                  fc=self.SURFACE_COLOUR,
                                  ec=self.LINE_COLOUR,
                                  lw=2,
                                  zorder=0)

        plt.gca().add_patch(table)

        for i in range(1, 3):
            line_x = plt.Line2D((i / 3 * self.LENGTH, i / 3 * self.LENGTH),
                                (0, self.WIDTH),
                                color=self.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            line_y = plt.Line2D((0, self.LENGTH),
                                (i / 3 * self.WIDTH, i / 3 * self.WIDTH),
                                color=self.LINE_COLOUR,
                                lw=0.75,
                                linestyle='--',
                                zorder=1)
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


class Sensor:
    """Represents a sensor.
    NOTE:   Not sure just how to represent coordinates yet,
            or if get_/set_coordinates() are necessary.
    """
    radius = 0.0035

    def __init__(self, coordinates: np.ndarray, name: str, plot: bool = True):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.name = name
        self.plot = plot

    def set_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def draw(self):
        """Draw the sensor."""
        sensor = plt.Circle(self.coordinates,
                            radius=self.radius,
                            fc='#AEAFA7',
                            ec='dimgray',
                            label='Sensor',
                            zorder=10)
        plt.gca().add_patch(sensor)

    def __str__(self):
        return self.name



class Actuator:
    """Represents an actuator.
    NOTE:   Not sure just how to represent coordinates yet,
            or if get_/set_coordinates() are necessary.
    """
    RADIUS = 0.005  # m

    def __init__(self, coordinates: np.ndarray, name: str = 'Actuator'):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.name = name

    def copy(self):
        """Define a function to copy the actuator"""
        return Actuator(self.coordinates)

    def set_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def draw(self):
        """Draw the actuator"""
        actuator = plt.Circle(self.coordinates,
                              radius=self.RADIUS,
                              fc='#D4434A',
                              ec='dimgray',
                              label=self.name,
                              zorder=10)
        plt.gca().add_patch(actuator)

    def __str__(self):
        return self.name



class MirroredSource(Actuator):
    """Represents a mirrored source."""
    def __init__(self, coordinates: np.ndarray):
        super().__init__(coordinates)

    def draw(self):
        """Draw the mirrored source."""
        mirrored_source = plt.Circle(self.coordinates,
                                     radius=self.RADIUS,
                                     fc='pink',
                                     ec='dimgray',
                                     label=f'Mirrored source',
                                     zorder=10)
        plt.gca().add_patch(mirrored_source)


class MirroredSensor(Sensor):
    """Represents a mirrored sensor."""
    def __init__(self, coordinates: np.ndarray, name: str):
        super().__init__(coordinates, name)

    def draw(self):
        """Draw the mirrored sensor."""
        mirrored_sensor = plt.Circle(self.coordinates,
                                     radius=self.radius,
                                     fc='white',
                                     ec='dimgray',
                                    #  label=f'Mirrored {self.name}',
                                     label='Mirrored sensor',
                                     zorder=10)
        plt.gca().add_patch(mirrored_sensor)
