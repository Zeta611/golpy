from functools import wraps
from PIL import Image
from scipy import signal
from typing import Callable, List, Tuple
import copy
import numpy as np
import time


def timeit(f):
    timeit.records = {}

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not DEBUG:
            return f(*args, **kwargs)
        name = f.__name__

        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        if name in timeit.records:
            timeit.records[name] += elapsed
        else:
            timeit.records[name] = elapsed
        return result

    return wrapper


@timeit
def progress(grid: np.ndarray) -> None:
    """Progress grid to the next generation."""

    # neighbor_cnts[y, x] is the number of live neighbors of grid[y, x].
    neighbor_cnts = signal.convolve2d(
        grid,
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype="uint8"),
        mode="same",
    )
    height, width = grid.shape
    old_grid = copy.deepcopy(grid)

    for y in range(height):
        for x in range(width):
            live_neighbors_cnt = neighbor_cnts[y, x]
            alive = old_grid[y, x]

            if alive and 2 <= live_neighbors_cnt <= 3:
                continue  # grid[y, x] stays alive
            elif not alive and live_neighbors_cnt == 3:
                grid[y, x] = 1
            else:
                grid[y, x] = 0


@timeit
def driver(
    grid: np.ndarray,
    handler: Callable[[np.ndarray, int], None],
    max_gen: int = 100,
) -> None:
    """Progresses the grid max_gen generations. Each generation of the grid is processed by the handler."""

    for gen in range(max_gen):
        handler(grid, gen)
        progress(grid)


def grid_print(grid: np.ndarray, generation: int) -> None:
    """Print the formatted grid on screen."""

    print(f"==== GEN {generation} ====")
    for row in grid:
        for cell in row:
            if cell:
                print("■", end="")
            else:
                print("□", end="")
        print()


@timeit
def parse_grid(
    text: str, size: Tuple[int, int], live: str = "*"
) -> np.ndarray:
    width, height = size
    grid = np.zeros((height, width), dtype="uint8")
    for i, line in enumerate(text.splitlines()):
        if i >= height:
            break
        for j, char in enumerate(line):
            if j >= width:
                break
            grid[i, j] = char == live
    return grid


@timeit
def add_grid_frame(grid: np.ndarray, generation: int) -> None:
    """Add the grid to the grid_frames"""

    arr_grid = enlarge_image(grid * 255, PIXELS_PER_CELL)
    image = Image.fromarray(arr_grid, mode="L").convert("P")
    grid_frames.append(image)


@timeit
def enlarge_image(image: np.ndarray, ratio: int) -> np.ndarray:
    """Enlarges each pixel in the image to a ratio * ratio square block."""

    return np.kron(image, np.ones((ratio, ratio), dtype="uint8"))


@timeit
def save_frames(grid_frames: List[Image.Image], filename: str) -> None:
    grid_frames[0].save(
        filename,
        save_all=True,
        append_images=grid_frames[1:],
        duration=DURATION,
        loop=0,
    )


DEBUG = False

WIDTH = 60
HEIGHT = 40
MAX_GEN = 300
PIXELS_PER_CELL = 10
DURATION = 50

glider_gun = parse_grid(
    """\
........................O
......................O.O
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO
OO........O...O.OO....O.O
..........O.....O.......O
...........O...O
............OO
""",
    size=(WIDTH, HEIGHT),
    live="O",
)

grid_frames: List[Image.Image] = []
driver(glider_gun, handler=add_grid_frame, max_gen=MAX_GEN)
save_frames(grid_frames, "glider_gun.gif")
print(timeit.records)
