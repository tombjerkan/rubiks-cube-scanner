# Rubik's Cube Scanner

Scans an image of a Rubik's Cube face to determine the colours on it.

## Prerequisites

- Python 3 (https://www.python.org/)
- Pipenv (https://pipenv.readthedocs.io/en/latest/)

## Installing

Install a Pipenv environment with all requirements:

```
pipenv install
```

## Usage

```
pipenv run ./run ./cube.png
```

See the script's help documentation for details on arguments:

```
pipenv run python scancube --help
```

## Code Style

The PEP 8 style guide (https://www.python.org/dev/peps/pep-0008/) is used. Automatically check code style using the `pycodestyle` tool:

```
pycodestyle .
```

## Built With

* [OpenCV](https://opencv.org/) - Computer vision library
