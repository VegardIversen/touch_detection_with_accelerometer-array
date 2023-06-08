"""Author: Niklas Strømsnes
Date: 2022-01-09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Table:
    """Represents the table and its edges."""

    """Table dimensions"""
    LENGTH = 0.80  # m
    WIDTH = 0.60  # m
    """Enum for representing edges"""
    TOP_EDGE = 1
    RIGHT_EDGE = 2
    BOTTOM_EDGE = 3
    LEFT_EDGE = 4
    """Colour settings for drawing"""
    SURFACE_COLOUR = "#fbe5b6"
    LINE_COLOUR = "#f0c18b"

    """A selection of table locations, in the middle of each block"""
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
        table = patches.Rectangle(
            (0, 0),
            self.LENGTH,
            self.WIDTH,
            fc=self.SURFACE_COLOUR,
            ec=self.LINE_COLOUR,
            lw=2,
            zorder=0,
        )

        plt.gca().add_patch(table)

        for i in range(1, 3):
            line_x = patches.Rectangle(
                (i / 3 * self.LENGTH, 0),
                0,
                self.WIDTH,
                linewidth=0.75,
                linestyle="--",
                edgecolor=self.LINE_COLOUR,
                facecolor="none",
                zorder=1,
            )
            line_y = patches.Rectangle(
                (0, i / 3 * self.WIDTH),
                self.LENGTH,
                0,
                linewidth=0.75,
                linestyle="--",
                edgecolor=self.LINE_COLOUR,
                facecolor="none",
                zorder=1,
            )
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


class Plate:
    """Represents the table and its edges."""

    """Table dimensions"""
    LENGTH = 1  # m
    WIDTH = 0.7  # m
    """Enum for representing edges"""
    TOP_EDGE = 1
    RIGHT_EDGE = 2
    BOTTOM_EDGE = 3
    LEFT_EDGE = 4
    """Colour settings for drawing"""
    SURFACE_COLOUR = "#f8f8f8"
    LINE_COLOUR = "lightgrey"

    """A selection of table locations, in the middle of each block"""
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
        table = patches.Rectangle(
            (0, 0),
            self.LENGTH,
            self.WIDTH,
            fc=self.SURFACE_COLOUR,
            ec=self.LINE_COLOUR,
            lw=2,
            zorder=0,
        )

        plt.gca().add_patch(table)

        for i in range(1, 3):
            line_x = patches.Rectangle(
                (i / 3 * self.LENGTH, 0),
                0,
                self.WIDTH,
                linewidth=0.75,
                linestyle="--",
                edgecolor=self.LINE_COLOUR,
                facecolor="none",
                zorder=1,
            )
            line_y = patches.Rectangle(
                (0, i / 3 * self.WIDTH),
                self.LENGTH,
                0,
                linewidth=0.75,
                linestyle="--",
                edgecolor=self.LINE_COLOUR,
                facecolor="none",
                zorder=1,
            )
            plt.gca().add_patch(line_x)
            plt.gca().add_patch(line_y)


class Sensor:
    """Represents a sensor.
    NOTE:   Not sure just how to represent coordinates yet,
            or if get_/set_coordinates() are necessary.
    """

    radius_m = 0.0035
    type_MEMS = False
    # If type_MEMS, use a square:
    edge_length_m = 0.003

    def __init__(
        self,
        coordinates: np.ndarray,
        name: str,
        plot: bool = True,
        type_MEMS: bool = False,
    ):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.name = name
        self.plot = plot
        self.type_MEMS = type_MEMS

    def set_coordinates(
        self,
        coordinates: np.ndarray,
    ):
        self.coordinates = coordinates
        self.x = coordinates[0]
        self.y = coordinates[1]

    def draw(self):
        """Draw the sensor as a square if type_MEMS is True, otherwise as a circle"""
        if self.type_MEMS:
            sensor = patches.Rectangle(
                (self.x - self.edge_length_m / 2, self.y - self.edge_length_m / 2),
                self.edge_length_m,
                self.edge_length_m,
                fc="#AEAFA7",
                ec="dimgray",
                label="Sensor",
                zorder=10,
            )
        else:
            sensor = plt.Circle(
                self.coordinates,
                radius=self.radius_m,
                fc="#AEAFA7",
                ec="dimgray",
                label="Sensor",
                zorder=10,
            )
        plt.gca().add_patch(sensor)

    def __str__(self):
        return self.name


class Actuator:
    """Represents an actuator"""

    RADIUS = 0.005  # m

    def __init__(self, coordinates: np.ndarray, name: str = "Actuator"):
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
        actuator = plt.Circle(
            self.coordinates,
            radius=self.RADIUS,
            fc="#D4434A",
            ec="darkred",
            label=self.name,
            zorder=10,
        )
        plt.gca().add_patch(actuator)

    def __str__(self):
        return self.name


class MirroredSource(Actuator):
    """Represents a mirrored source, to use when drawing reflection sources"""

    def __init__(self, coordinates: np.ndarray):
        super().__init__(coordinates)

    def draw(self):
        """Draw the mirrored source."""
        mirrored_source = plt.Circle(
            self.coordinates,
            radius=self.RADIUS,
            fc="pink",
            ec="dimgray",
            label="Mirrored source",
            zorder=10,
        )
        plt.gca().add_patch(mirrored_source)


class MirroredSensor(Sensor):
    """Represents a mirrored sensor, to use when drawing reflection sources"""

    def __init__(self, coordinates: np.ndarray, name: str):
        super().__init__(coordinates, name)

    def draw(self):
        """Draw the mirrored sensor."""
        mirrored_sensor = plt.Circle(
            self.coordinates,
            radius=self.radius_m,
            fc="white",
            ec="dimgray",
            #  label=f'Mirrored {self.name}',
            label="Mirrored sensor",
            zorder=10,
        )
        plt.gca().add_patch(mirrored_sensor)
