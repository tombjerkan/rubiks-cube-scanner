# Rubik's Cube Scanner

Scans an image of a Rubik's Cube face to determine the colours on it.

## Prerequisites

- Python 3.6 (https://www.python.org/)
- Poetry (https://poetry.eustace.io/)
- Pyenv, for handling different versions of python (https://github.com/pyenv/pyenv)

## Installing

Ensure the correct version of Python (3.6) is activated:

```
pyenv install 3.6.9
pyenv global 3.6.9
```

Install a virtual environment and all requirements:

```
poetry install
```

## Usage

```
poetry run ./run <path-to-image-of-cube-face>
```

See the script's help documentation for details on arguments:

```
poetry run ./run --help
```

## Examples

To run on the example file:

```
poetry run ./run ./cube.png
```

To run on the example file with all intermediate images output:

```
poetry run ./run ./cube.png \
    --edges ./1-edges.png \
    --lines ./2-lines.png \
    --orth ./3-orth.png \
    --comb ./4-comb.png \
    --clines ./5-clines.png \
    --cpoints ./6-cpoints.png
```

## Code Style

The PEP 8 style guide (https://www.python.org/dev/peps/pep-0008/) is used. Automatically check code style using the `pycodestyle` tool:

```
poetry run pycodestyle .
```

## Built With

* [OpenCV](https://opencv.org/) - Computer vision library
