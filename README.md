# py-game-of-life
Efficient Conway's Game of Life implemented in Python using NumPy.

## Example Output
The following GIF can be generated using the command:
```sh
python game-of-life.py --demo glidergun --out glider_gun.gif --ppc 10 --pos TL -W60 -H40
```

![The Gosper Glider Gun](glider_gun.gif)

## Usage
```
usage: game-of-life.py [-h] (-i IN | -d DEMO) [-o OUT | --debug-print]
                       [-W WIDTH] [-H HEIGHT] [-M MAX_GEN] [--ppc PPC]
                       [-P POS] [-p]

optional arguments:
  -h, --help            show this help message and exit
  -i IN, --in IN
  -d DEMO, --demo DEMO
  -o OUT, --out OUT
  --debug-print

  -W WIDTH, --width WIDTH
  -H HEIGHT, --height HEIGHT

  -M MAX_GEN, --max-gen MAX_GEN
  --ppc PPC
  -P POS, --pos POS

  -p, --profile
```

## Input Format
```
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
