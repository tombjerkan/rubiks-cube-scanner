#!/usr/bin/env python3

from collections import namedtuple
import cv2
import numpy as np
import math


IntermediateImageSet = namedtuple(
    "IntermediateImageSet",
    ["edges", "lines", "orthogonal_lines", "combined_lines", "centre_lines",
     "centre_points"]
)


def scan(cube_image):
    """Scan the given cube image and return the colours of the cube face.

    Returns tuple (colours, intermediate images) where colours is a tuple
    of scanned colours and intermediate images are images generated through
    the scanning process to be used for debugging.
    """
    edge_image = _detect_edges(cube_image)
    lines = _detect_lines(edge_image)
    orthogonal_lines = _horizontal_and_vertical_lines(lines)

    combined_lines = _combine_lines(orthogonal_lines)
    if len(combined_lines) != 8:
        return None

    centre_lines = _find_centre_lines(combined_lines)
    if centre_lines is None:
        return None

    centre_points = _find_centres(centre_lines)

    square_colours = _square_colours(cube_image, centre_points)
    rubiks_colours = [_to_rubiks_colour(colour) for colour in square_colours]

    intermediate_images = IntermediateImageSet(
        edges=edge_image,
        lines=_draw_lines(cube_image, lines),
        orthogonal_lines=_draw_lines(cube_image, orthogonal_lines),
        combined_lines=_draw_lines(cube_image, combined_lines),
        centre_lines=_draw_lines(
            cube_image,
            (line for line_group in centre_lines for line in line_group)
        ),
        centre_points=_draw_points(cube_image, centre_points)
    )

    return rubiks_colours, intermediate_images


def _detect_edges(image):
    """Detects edges in the given image using Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edge_image = cv2.Canny(blurred_gray_image, 0, 50)

    return edge_image


def _detect_lines(edge_image, threshold=125):
    """Detects lines in the given edge image using the Hough transform."""
    lines = cv2.HoughLines(edge_image, 1, np.pi/180, threshold)
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
    image_copy = image.copy()

    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*a)
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*a)

        cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image_copy


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


def _similar(line_1, line_2, rho_threshold=50, theta_threshold=np.pi/18):
    """Returns True if the two lines are similar enough, False otherwise."""
    rho_1, theta_1 = line_1
    rho_2, theta_2 = line_2

    if abs(rho_1 - rho_2) > rho_threshold:
        return False

    if abs(theta_1 - theta_2) > theta_threshold:
        return False

    return True


def _average_line(lines):
    """Combines a group of lines and returns their average."""
    average_rho = sum(rho for rho, _ in lines) / len(lines)
    average_theta = sum(theta for _, theta in lines) / len(lines)
    return average_rho, average_theta


def _combine_lines(lines):
    """Combines lines that are similar to one another."""
    similar_line_groups = [
        # Convert to tuple to be hashable for set below
        tuple([line for line in lines if _similar(line, original_line)])
        for original_line in lines
    ]

    unique_similar_line_groups = set(similar_line_groups)

    combined_lines = {
        _average_line(similar_lines)
        for similar_lines in unique_similar_line_groups
    }

    return combined_lines


def _identify_horizontal_lines(lines):
    """Identifies the horizontal lines and returns them from top to bottom."""
    horizontal_lines = [line for line in lines if _is_horizontal(line)]
    # Rho shows how far away from origin therefore position in top to bottom
    horizontal_lines.sort(key=lambda line: line[0])
    return tuple(horizontal_lines)


def _identify_vertical_lines(lines):
    """Identifies the vertical lines and returns them from left to right."""
    vertical_lines = [line for line in lines if _is_vertical(line)]
    # Rho shows how far away from origin therefore position in left to right
    vertical_lines.sort(key=lambda line: line[0])
    return tuple(vertical_lines)


def _find_centre_lines(lines):
    """Finds the lines passing through the centre of each square"""
    # Number identifies position from 1 to 4 in top to bottom ordering
    horizontal_lines = _identify_horizontal_lines(lines)
    if len(horizontal_lines) != 4:
        return None
    hor_1, hor_2, hor_3, hor_4 = horizontal_lines

    # Number identifies position from 1 to 4 in left to right ordering
    vertical_lines = _identify_vertical_lines(lines)
    if len(vertical_lines) != 4:
        return None
    vert_1, vert_2, vert_3, vert_4 = vertical_lines

    top_horizontal = _average_line([hor_1, hor_2])
    middle_horizontal = _average_line([hor_2, hor_3])
    bottom_horizontal = _average_line([hor_3, hor_4])

    left_vertical = _average_line([vert_1, vert_2])
    middle_vertical = _average_line([vert_2, vert_3])
    right_vertical = _average_line([vert_3, vert_4])

    centre_horizontals = (top_horizontal, middle_horizontal, bottom_horizontal)
    centre_verticals = (left_vertical, middle_vertical, right_vertical)

    return (centre_horizontals, centre_verticals)


def _intersection(line_1, line_2):
    """Finds the (x, y) point where two lines intersect."""
    rho_1, theta_1 = line_1
    rho_2, theta_2 = line_2

    cos_theta_1 = math.cos(theta_1)
    sin_theta_1 = math.sin(theta_1)
    cos_theta_2 = math.cos(theta_2)
    sin_theta_2 = math.sin(theta_2)

    det = cos_theta_1*sin_theta_2 - sin_theta_1*cos_theta_2

    # det is None when lines are parallel
    if det is None:
        return None

    x = (sin_theta_2*rho_1 - sin_theta_1*rho_2) / det
    y = (cos_theta_1*rho_2 - cos_theta_2*rho_1) / det

    return (x, y)


def _find_centres(centre_lines):
    horizontal_lines, vertical_lines = centre_lines

    top_left = _intersection(horizontal_lines[0], vertical_lines[0])
    top_middle = _intersection(horizontal_lines[0], vertical_lines[1])
    top_right = _intersection(horizontal_lines[0], vertical_lines[2])

    middle_left = _intersection(horizontal_lines[1], vertical_lines[0])
    middle = _intersection(horizontal_lines[1], vertical_lines[1])
    middle_right = _intersection(horizontal_lines[1], vertical_lines[2])

    bottom_left = _intersection(horizontal_lines[2], vertical_lines[0])
    bottom_middle = _intersection(horizontal_lines[2], vertical_lines[1])
    bottom_right = _intersection(horizontal_lines[2], vertical_lines[2])

    return (
        top_left, top_middle, top_right,
        middle_left, middle, middle_right,
        bottom_left, bottom_middle, bottom_right
    )


def _draw_points(image, points):
    """Returns a copy of the image with the given points drawn on."""
    image_copy = image.copy()
    for x, y in points:
        cv2.circle(image_copy, (int(x), int(y)), 3, (255, 0, 255), -1)

    return image_copy


def _colours_around_centre(image, centre_point, offset):
    """Returns a list of colours in a square around a centre point"""
    centre_x, centre_y = centre_point
    left_x = int(centre_x) - offset
    right_x = int(centre_x) + offset
    top_y = int(centre_y) - offset
    bottom_y = int(centre_y) + offset

    pixel_colours = [
        image[y][x]
        for x in range(left_x, right_x + 1)
        for y in range(top_y, bottom_y + 1)
    ]

    return pixel_colours


def _average_colour(colours):
    """Find the average of a list of colours."""
    b_average = math.sqrt(sum(b**2 for b, _, _ in colours) / len(colours))
    g_average = math.sqrt(sum(g**2 for _, g, _ in colours) / len(colours))
    r_average = math.sqrt(sum(r**2 for _, _, r in colours) / len(colours))

    return (b_average, g_average, r_average)


def _square_colours(image, centre_points, offset=20):
    """Finds colour of each square using square around its centre point."""
    square_colours = tuple(
        _average_colour(_colours_around_centre(image, centre_point, offset))
        for centre_point in centre_points
    )

    return square_colours


def _colour_similarity(colour_1, colour_2):
    """Returns the similarity between the colours from 0 (none) to 100 (the
    same).
    """
    b_1, g_1, r_1 = colour_1
    b_2, g_2, r_2 = colour_2

    diff_blue = abs(b_1 - b_2)
    diff_green = abs(g_1 - g_2)
    diff_red = abs(r_1 - r_2)

    difference = (diff_blue + diff_green + diff_red) / 3 / 255 * 100
    similarity = 100 - difference

    return similarity


def _to_rubiks_colour(colour):
    """Returns the nearest rubiks cube colour to the given colour."""
    rubiks_colours = [
        ('white', (255., 255., 255.)),
        ('green', (72., 155., 0.)),
        ('red', (52., 18., 183.)),
        ('blue', (173., 70., 0.)),
        ('orange', (0., 88., 255.)),
        ('yellow', (0., 213., 255.))
    ]

    similarities = [
        (rubiks_colour_name, _colour_similarity(colour, rubiks_colour))
        for (rubiks_colour_name, rubiks_colour) in rubiks_colours
    ]

    similarities.sort(key=lambda similarity: similarity[1], reverse=True)

    return similarities[0][0]
