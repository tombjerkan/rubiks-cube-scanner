import cv2


def scan_cube(cubeImage, edgeImageFile=None):
    """Scan the given cube image and return the colours of the cube face.

    Returns None if cube face could not be scanned.
    Saves the intermediate edge image generated to the filename given.
    """
    edgeImage = _detect_edges(cubeImage)
    if edgeImageFile is not None:
        cv2.imwrite(edgeImageFile, edgeImage)

    return None


def _detect_edges(image):
    """Detects edges in the given image using Canny edge detection."""
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurredGrayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    edgeImage = cv2.Canny(blurredGrayImage, 0, 50)

    return edgeImage
