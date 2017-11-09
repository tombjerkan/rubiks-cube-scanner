import cv2
import numpy as np


def scan_cube(
        cubeImage,
        edgeImageFile=None,
        linesImageFile=None,
        orthogonalLinesImageFile=None,
        combinedLinesImageFile=None):
    """Scan the given cube image and return the colours of the cube face.

    Returns None if cube face could not be scanned.
    Saves the intermediate edge image generated to the filename given.
    Saves the intermediate lines image generated to the filename given.
    Saves the intermediate orthogonal lines image generated to the filename
    given.
    Saves the intermediate combined lines image generated to the filename
    given.
    """
    edgeImage = _detect_edges(cubeImage)
    if edgeImageFile is not None:
        cv2.imwrite(edgeImageFile, edgeImage)

    lines = _detect_lines(edgeImage)
    if linesImageFile is not None:
        linesImage = _draw_lines(cubeImage, lines)
        cv2.imwrite(linesImageFile, linesImage)

    orthogonalLines = _horizontal_and_vertical_lines(lines)
    if orthogonalLinesImageFile is not None:
        orthogonalLinesImage = _draw_lines(cubeImage, orthogonalLines)
        cv2.imwrite(orthogonalLinesImageFile, orthogonalLinesImage)

    combinedLines = _combine_lines(orthogonalLines)
    if combinedLinesImageFile is not None:
        combinedLinesImage = _draw_lines(cubeImage, combinedLines)
        cv2.imwrite(combinedLinesImageFile, combinedLinesImage)

    return None


def _detect_edges(image):
    """Detects edges in the given image using Canny edge detection."""
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredGrayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    edgeImage = cv2.Canny(blurredGrayImage, 0, 50)

    return edgeImage


def _detect_lines(edgeImage, threshold=125):
    """Detects lines in the given edge image using the Hough transform."""
    lines = cv2.HoughLines(edgeImage, 1, np.pi/180, threshold)
    # Change line representation from [(rho, theta)] to just (rho, theta)
    lines = [line[0] for line in lines]

    # Stop rho from wrapping round from positive to negative and theta from
    # wrapping round from pi to 0. This ensures similar lines have similar
    # values when compared.
    lines = [
        (rho, theta) if rho >= 0 else (-rho, theta - np.pi)
        for (rho, theta) in lines
    ]

    return lines


def _draw_lines(image, lines):
    """Returns a copy of the image with the given lines drawn on."""
    imageCopy = image.copy()

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(imageCopy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return imageCopy


def _is_horizontal(line):
    """Returns True if line is within 1/36pi of horizontal."""
    _, theta = line
    return theta > np.pi*17/36 and theta < np.pi*19/36


def _is_vertical(line):
    """Returns True if line is within 1/36pi of vertical."""
    _, theta = line
    return theta > -np.pi/36 and theta < np.pi/36


def _horizontal_and_vertical_lines(lines):
    """Returns only horizontal and vertical lines of the lines given."""
    return [
        line for line in lines
        if _is_horizontal(line) or _is_vertical(line)
    ]


def _similar(line1, line2, rhoThreshold=50, thetaThreshold=np.pi/18):
    """Returns True if the two lines are similar enough, False otherwise."""
    rho1, theta1 = line1
    rho2, theta2 = line2

    if abs(rho1 - rho2) > rhoThreshold:
        return False

    if abs(theta1 - theta2) > thetaThreshold:
        return False

    return True


def _average_line(lines):
    """Combines a group of lines and returns their average."""
    averageRho = sum(rho for rho, _ in lines) / len(lines)
    averageTheta = sum(theta for _, theta in lines) / len(lines)
    return averageRho, averageTheta


def _combine_lines(lines):
    """Combines lines that are similar to one another."""
    similarLineGroups = [
        # Convert to tuple to be hashable for set below
        tuple([line for line in lines if _similar(line, originalLine)])
        for originalLine in lines
    ]

    uniqueSimilarLineGroups = set(similarLineGroups)

    combinedLines = {
        _average_line(similarLines)
        for similarLines in uniqueSimilarLineGroups
    }

    return combinedLines
