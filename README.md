# GoLPy

[![GitHub
license](https://img.shields.io/github/license/Zeta611/golpy?style=flat-square)](https://github.com/Zeta611/golpy/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/golpy?style=flat-square)](https://pypi.org/project/golpy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

GoLPy is an efficient Conway's Game of Life implemented in Python using NumPy.

## Example Output

The following GIF can be generated using the command:

```sh
life --demo glidergun --out glider_gun.gif --ppc 10 --pos TL -W60 -H40
```

![The Gosper Glider Gun](glider_gun.gif)

## Installation

```sh
pip install golpy
```

## Usage

```sh
usage: life [-h] (-i GRID_INPUT | -d DEMO) [-o FILE | --debug-print]
            [-W WIDTH] [-H HEIGHT] [-M MAX_GEN] [--ppc PIXELS] [-P POSITION]
            [-p]

optional arguments:
  -h, --help            show this help message and exit
  -i GRID_INPUT, --in GRID_INPUT
                        Parse the initial grid from <GRID_INPUT>
  -d DEMO, --demo DEMO  Try one of the provided demos: one of 'glidergun' and
                        'glidergen'
  -o FILE, --out FILE   Place the output into <FILE>
  --debug-print         Print the generated frames directly to the terminal,
                        instead of saving them

  -W WIDTH, --width WIDTH
                        Width of the grid
  -H HEIGHT, --height HEIGHT
                        Height of the grid

  -M MAX_GEN, --max-gen MAX_GEN
                        Number of generations to simulate
  --ppc PIXELS          Set the width and the height of each cell to <PIXELS>
  -P POSITION, --pos POSITION
                        One of 'C', 'T', 'B', 'L', 'R', 'TL', 'TR', 'BL', and
                        'BR'

  -p, --profile         Measure the performance
```

To use without installing,

```sh
python -m golpy # ...
```

## Input Format

```txt
........................O
......................O.O
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO
OO........O...O.OO....O.O
..........O.....O.......O
...........O...O
............OO
```

Use `.` for a dead cell, `O` (`chr(79)`) for a live cell.

## License

[MIT](LICENSE)
