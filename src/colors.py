import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_hex_code(rgb_px):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_px[0]), int(rgb_px[1]), int(rgb_px[2]))


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def apply_mask(img):
    """
    Remove background colors from an input image with Canny Edge Detection
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=-1)

    masked_image = cv2.bitwise_and(img, img, mask=mask)
    masked_pixels = masked_image[mask == 255]
    return masked_pixels


def retrieve_dominant_colors(img, k=5, mask=True):
    """
    Use k-Nearest Neighbors (KNN) to retrieve the dominant colors within a bounding
    box or an input image.

    You can pass a `mask` argument where if `True`, this function will
    attempt to ignore background colors.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    pixels = apply_mask(img) if mask else img.reshape(-1, 3)

    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_colors = kmeans.cluster_centers_
    percentages = (counts / counts.sum()) * 100

    color_info = []

    for idx, color in enumerate(dominant_colors):
        hex_code = get_hex_code(color)
        percentage = percentages[idx]
        color_info.append((hex_code, percentage))

    return color_info


def get_color_luminance(hex_code):
    red, green, blue = hex_to_rgb(hex_code)

    # color brightness formula based from the RGB to YIQ conversion formula.
    # see: https://www.w3.org/TR/AERT/#color-contrast
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    return luminance
