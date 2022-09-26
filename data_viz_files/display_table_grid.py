from __future__ import division
import sys
import pygame
from pygame.locals import KEYDOWN, K_q

# CONSTANTS:
SCREENSIZE = WIDTH, HEIGHT = 325, 325
BLACK = (0, 0, 0)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
BLUE = (135, 206, 250)
CONTAINER_WIDTH_HEIGHT = 300  # Not to be confused with SCREENSIZE
CONT_X, CONT_Y = 10, 10  # TOP LEFT OF CONTAINER
DIVISIONS = 3
# VARS:
_VARS = {'surf': False}


# config
cell_to_color = {"x": 1, "y": 1}
confidence_score = 1
cell_color = (255 * (1 - confidence_score), 255, 255 * (1 - confidence_score))


def main():
    pygame.init()
    _VARS['surf'] = pygame.display.set_mode(size=SCREENSIZE)
    while True:
        checkEvents()
        _VARS['surf'].fill(color=WHITE)
        drawGrid(divisions=DIVISIONS)
        drawRect(divisions=DIVISIONS, row=cell_to_color["x"], col=cell_to_color["y"])
        pygame.display.update()


def drawRect(divisions, row, col):
    # Draw a box in the cell specified by row and col
    MARKER_PADDING = 0.2
    pygame.draw.rect(
        _VARS['surf'], cell_color,
        (CONT_X + row * CONTAINER_WIDTH_HEIGHT / divisions 
            + (MARKER_PADDING * CONTAINER_WIDTH_HEIGHT / divisions),
        CONT_Y + col * CONTAINER_WIDTH_HEIGHT / divisions
            + (MARKER_PADDING * CONTAINER_WIDTH_HEIGHT / divisions),
        (1 - 2 * MARKER_PADDING) * CONTAINER_WIDTH_HEIGHT / divisions,
        (1 - 2 * MARKER_PADDING) * CONTAINER_WIDTH_HEIGHT / divisions)
    )


def drawGrid(divisions):

    # Get cell size, just one since its a square grid.
    cellSize = CONTAINER_WIDTH_HEIGHT / divisions

    # DRAW Grid Border:
    # TOP lEFT TO RIGHT
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (CONT_X, CONT_Y),
      (CONTAINER_WIDTH_HEIGHT + CONT_X, CONT_Y), 2)
    # # BOTTOM lEFT TO RIGHT
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (CONT_X, CONTAINER_WIDTH_HEIGHT + CONT_Y),
      (CONTAINER_WIDTH_HEIGHT + CONT_X, CONTAINER_WIDTH_HEIGHT + CONT_Y), 2)
    # # LEFT TOP TO BOTTOM
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (CONT_X, CONT_Y),
      (CONT_X, CONT_Y + CONTAINER_WIDTH_HEIGHT), 2)
    # # RIGHT TOP TO BOTTOM
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (CONTAINER_WIDTH_HEIGHT + CONT_X, CONT_Y),
      (CONTAINER_WIDTH_HEIGHT + CONT_X, CONTAINER_WIDTH_HEIGHT + CONT_Y), 2)


    # VERTICAL DIVISIONS: (0,1,2) for grid(3) for example
    for x in range(divisions):
        pygame.draw.line(
           _VARS['surf'], BLACK,
           (CONT_X + (cellSize * x), CONT_Y),
           (CONT_X + (cellSize * x), CONTAINER_WIDTH_HEIGHT + CONT_Y), 2)
    # # HORIZONTAl DIVISIONS
        pygame.draw.line(
          _VARS['surf'], BLACK,
          (CONT_X, CONT_Y + (cellSize * x)),
          (CONT_X + CONTAINER_WIDTH_HEIGHT, CONT_Y + (cellSize * x)), 2)


def checkEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_q:
            pygame.quit()
            sys.exit()


if __name__ == '__main__':
    main()
