import argparse
import functools
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

from .util import timeit


@timeit()
def count_neighbors(grid: np.ndarray) -> np.ndarray:
    #  neighbor_cnts[y, x] is the number of live neighbors of grid[y, x].
    #  scipy.signal.convolve2d() can be used, but manually adding is faster than using convolution.
    neighbor_cnts = np.empty(grid.shape, dtype="uint8")
    # fmt:off
    # Inner area
    neighbor_cnts[1:-1, 1:-1] = (
        grid[:-2, :-2]  # top-left
        + grid[:-2, 1:-1]  # top
        + grid[:-2, 2:]  # top right
        + grid[1:-1, 2:]  # right
        + grid[1:-1, :-2]  # left
        + grid[2:, 2:]  # bottom-right
        + grid[2:, 1:-1]  # bottom
        + grid[2:, :-2]  # bottom-left
    )
    # Four corners
    neighbor_cnts[0, 0] = (  # top-left
        grid[0, 1]  # right
        + grid[1, 1]  # bottom-right
        + grid[1, 0]  # bottom
    )
    neighbor_cnts[0, -1] = (  # top-right
        grid[1, -1]  # bottom
        + grid[1, -2]  # bottom-left
        + grid[0, -2]  # left
    )
    neighbor_cnts[-1, -1] = (  # bottom-right
        grid[-1, -2]  # left
        + grid[-2, -2]  # top-left
        + grid[-2, -1]  # top
    )
    neighbor_cnts[-1, 0] = (  # bottom-left
        grid[-2, 0]  # top
        + grid[-2, 1]  # top-right
        + grid[-1, 1]  # right
    )
    # Four edges
    neighbor_cnts[0, 1:-1] = (  # top
        grid[0, 2:]  # right
        + grid[1, 2:]  # bottom-right
        + grid[1, 1:-1]  # bottom
        + grid[1, :-2]  # bottom-left
        + grid[0, :-2]  # left
    )
    neighbor_cnts[1:-1, -1] = (  # right
        grid[2:, -1]  # bottom
        + grid[2:, -2]  # bottom-left
        + grid[1:-1, -2]  # left
        + grid[:-2, -2]  # top-left
        + grid[:-2, -1]  # top
    )
    neighbor_cnts[-1, 1:-1] = (  # bottom
        grid[-1, :-2]  # left
        + grid[-2, :-2]  # top-left
        + grid[-2, 1:-1]  # top
        + grid[-2, 2:]  # top-right
        + grid[-1, 2:]  # right
    )
    neighbor_cnts[1:-1, 0] = (  # left
        grid[:-2, 0]  # top
        + grid[:-2, 1]  # top-right
        + grid[1:-1, 1]  # right
        + grid[2:, 1]  # bottom-right
        + grid[2:, 0]  # bottom
    )
    # fmt:on
    return neighbor_cnts


@timeit()
def progress(grid: np.ndarray) -> None:
    """Progress grid to the next generation."""

    neighbor_cnts = count_neighbors(grid)

    grid_v = grid.ravel()
    neighbor_cnt_v = neighbor_cnts.ravel()

    birth_rule = (grid_v == 0) & (neighbor_cnt_v == 3)
    survive_rule = (grid_v == 1) & ((neighbor_cnt_v == 2) | (neighbor_cnt_v == 3))

    grid_v[...] = 0
    grid_v[birth_rule | survive_rule] = 1


@timeit()
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


@timeit()
def parse_grid(
    text: str, size: Tuple[int, int], pos: str = "C", live: str = "O"
) -> np.ndarray:
    lines = textwrap.dedent(text).strip().splitlines()
    text_width = max(len(line) for line in lines)
    text_height = len(lines)

    width, height = size
    if width < text_width or height < text_height:
        raise ValueError(
            f"given text of size {(text_width, text_height)} larger than grid size {size}"
        )

    grid = np.zeros((height, width), dtype="uint8")

    pos_idx: Dict[str, Tuple[int, int]] = {
        "C": (height // 2 - text_height // 2, width // 2 - text_width // 2),
        "T": (0, width // 2 - text_width // 2),
        "B": (height - text_height, width // 2 - text_width // 2),
        "L": (height // 2 - text_height // 2, 0),
        "R": (height // 2 - text_height // 2, width - text_width),
        "TL": (0, 0),
        "TR": (0, width - text_width),
        "BL": (height - text_height, 0),
        "BR": (height - text_height, width - text_width),
    }
    offset = pos_idx[pos.upper()]

    for i, line in enumerate(lines):
        if i >= height:
            break
        for j, char in enumerate(line):
            if j >= width:
                break
            grid[i + offset[0], j + offset[1]] = char == live
    return grid


@timeit()
def add_grid_frame(
    grid: np.ndarray,
    generation: int,
    grid_frames: List[np.ndarray],
    pixels_per_cell: int,
) -> None:
    """Add the grid to the grid_frames"""

    arr_grid = enlarge_image(grid * 255, pixels_per_cell)
    image = Image.fromarray(arr_grid, mode="L").convert("P")
    grid_frames.append(image)


@timeit()
def enlarge_image(image: np.ndarray, ratio: int) -> np.ndarray:
    """Enlarges each pixel in the image to a ratio * ratio square block."""

    return np.kron(image, np.ones((ratio, ratio), dtype="uint8"))


@timeit()
def save_frames(
    grid_frames: List[Image.Image], filename: str, duration: int = 50
) -> None:
    grid_frames[0].save(
        filename,
        save_all=True,
        append_images=grid_frames[1:],
        duration=duration,
        loop=0,
    )


def get_demo(name: str, size: Tuple[int, int], pos: str = "C") -> np.ndarray:
    if name == "random":
        return np.random.randint(0, 2, size, dtype="uint8")

    demos = {
        "glidergun": lambda: parse_grid(
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
            size,
            pos,
        ),
        "glidergen": lambda: parse_grid(
            """\
    ....OOOO

    ..OOOOOOOO

    OOOOOOOOOOOO

    ..OOOOOOOO

    ....OOOO
    """,
            size,
            pos,
        ),
    }

    return demos[name]()


def main() -> None:
    # Setup command line options

    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--in",
        metavar="GRID_INPUT",
        help="Parse the initial grid from <GRID_INPUT>",
    )
    input_group.add_argument(
        "-d",
        "--demo",
        help="Try one of the provided demos: one of 'glidergun' and 'glidergen'",
    )

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--out",
        default="out.gif",
        metavar="FILE",
        help="Place the output into <FILE>",
    )
    output_group.add_argument(
        "--debug-print",
        action="store_true",
        help="Print the generated frames directly to the terminal, instead of saving them",
    )

    size_group = parser.add_argument_group()
    size_group.add_argument(
        "-W", "--width", type=int, default=100, help="Width of the grid"
    )
    size_group.add_argument(
        "-H", "--height", type=int, default=100, help="Height of the grid"
    )

    option_group = parser.add_argument_group()
    option_group.add_argument(
        "-M",
        "--max-gen",
        type=int,
        default=300,
        help="Number of generations to simulate",
    )
    option_group.add_argument(
        "--ppc",
        type=int,
        default=1,
        metavar="PIXELS",
        help="Set the width and the height of each cell to <PIXELS>",
    )
    option_group.add_argument(
        "-P",
        "--pos",
        default="C",
        metavar="POSITION",
        help="One of 'C', 'T', 'B', 'L', 'R', 'TL', 'TR', 'BL', and 'BR'",
    )

    dev_group = parser.add_argument_group()
    dev_group.add_argument(
        "-p", "--profile", action="store_true", help="Measure the performance"
    )

    args = parser.parse_args()

    filename = args.out
    size = (args.width, args.height)
    max_gen = args.max_gen
    ppc = args.ppc
    pos = args.pos
    DURATION = 50

    timeit.on = args.profile

    if in_file := getattr(args, "in"):
        grid = parse_grid(Path(in_file).read_text(), size, pos)
    else:  # demo mode
        grid = get_demo(args.demo, size, pos)

    # Run Game of Life
    grid_frames: List[Image.Image] = []

    if args.debug_print:
        driver(grid, handler=grid_print, max_gen=max_gen)
    else:
        driver(
            grid,
            handler=functools.partial(
                add_grid_frame, grid_frames=grid_frames, pixels_per_cell=ppc
            ),
            max_gen=max_gen,
        )
        save_frames(grid_frames, filename, DURATION)

    if timeit.on:
        print(timeit.records)


if __name__ == "__main__":
    main()
