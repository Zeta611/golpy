from nptyping import NDArray
from PIL import Image
from typing import Any, Callable, List
import copy
import numpy as np
import textwrap


def count_live_neighbors(grid: List[List[bool]], x: int, y: int) -> int:
    """Returns the number of live neighbors of grid[y][x]. Neighbors out of the boundaries of the grid are considered dead."""

    width = len(grid[0])
    height = len(grid)

    dxs = [-1, 0, 1]
    dys = [-1, 0, 1]

    cnt = 0

    for dx in dxs:
        for dy in dys:
            if dx == dy == 0:
                continue
            if x + dx < 0 or x + dx >= width or y + dy < 0 or y + dy >= height:
                continue
            cnt += grid[y + dy][x + dx]
    return cnt


def progress(grid: List[List[bool]]) -> None:
    """Progress grid to the next generation."""

    width = len(grid[0])
    height = len(grid)

    old_grid = copy.deepcopy(grid)

    for y in range(height):
        for x in range(width):
            live_neighbors_cnt = count_live_neighbors(old_grid, x, y)
            alive = old_grid[y][x]

            if alive and 2 <= live_neighbors_cnt <= 3:
                continue  # grid[y][x] stays alive
            elif not alive and live_neighbors_cnt == 3:
                grid[y][x] = True
            else:
                grid[y][x] = False


def driver(
    grid: List[List[bool]],
    handler: Callable[[List[List[bool]], int], None],
    max_gen: int = 100,
) -> None:
    """Progresses the grid max_gen generations. Each generation of the grid is processed by the handler."""

    for gen in range(max_gen):
        handler(grid, gen)
        progress(grid)


def grid_print(grid: List[List[bool]], generation: int) -> None:
    """Print the formatted grid on screen."""

    print(f"==== GEN {generation} ====")
    for row in grid:
        for cell in row:
            if cell:
                print("■", end="")
            else:
                print("□", end="")
        print()


def parse_grid(text: str) -> List[List[bool]]:
    return [[char == "*" for char in line] for line in text.splitlines()]


def add_grid_frame(grid: List[List[bool]], generation: int) -> None:
    """Add the grid to the grid_frames"""

    arr_grid = enlarge_image(
        np.array(grid, dtype="uint8") * 255, PIXELS_PER_CELL
    )
    image = Image.fromarray(arr_grid, mode="L").convert("P")
    grid_frames.append(image)


def enlarge_image(
    image: NDArray[NDArray[int]], ratio: int
) -> NDArray[NDArray[int]]:
    enlarged = np.empty(
        shape=(image.shape[0] * ratio, image.shape[1] * ratio),
        dtype=image.dtype,
    )
    for i in range(enlarged.shape[0]):
        for j in range(enlarged.shape[1]):
            enlarged[i][j] = image[i // ratio][j // ratio]
    return enlarged


def save_frames(grid_frames: List[Image.Image]) -> None:
    grid_frames[0].save(
        "glider.gif",
        save_all=True,
        append_images=grid_frames[1:],
        duration=DURATION,
        loop=0,
    )


N = 40
MAX_GEN = 120
PIXELS_PER_CELL = 10
DURATION = 50

grid = [[False] * N for _ in range(N)]
grid[1][2] = True
grid[2][3] = True
grid[3][1] = True
grid[3][2] = True
grid[3][3] = True

grid_frames: List[Image.Image] = []
driver(grid, handler=add_grid_frame, max_gen=MAX_GEN)
save_frames(grid_frames)
