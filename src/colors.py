import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial import distance
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import slic
from skimage.util import img_as_float

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


def retrieve_dominant_colors_canny(img, k=5, mask=True):
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


def merge_similar_colors(cluster_centers, threshold=25):
    merged = []
    used = set()

    for i, color_1 in enumerate(cluster_centers):
        if i in used:
            continue

        similar_group = [color_1]
        
        for j, color_2 in enumerate(cluster_centers[i + 1:], start=i + 1):
            if j in used:
                continue

            # use Euclidean distance to estimate color similarity between 2 LAB colors
            if distance.euclidean(color_1, color_2) < threshold:
                similar_group.append(color_2)
                used.add(j)

        merged_color = np.mean(similar_group, axis=0)
        merged.append(merged_color)

    return np.array(merged)


def retrieve_dominant_colors_lab(img, threshold=25, k=5):
    """
    Uses KNN to retrieve dominant colors within an input image, but uses the CIELAB
    color space instead of RGB.

    You can pass a `mask` argument where if `True`, this function will
    attempt to ignore background colors.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = lab_img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixels)

    cluster_centers = kmeans.cluster_centers_
    merged_colors = merge_similar_colors(cluster_centers, threshold=threshold)
    merged_rgb = cv2.cvtColor(np.uint8([merged_colors]), cv2.COLOR_LAB2BGR)[0]

    new_counts = np.array([np.sum(labels == i) for i in range(len(merged_colors))])
    percentages = (new_counts / new_counts.sum()) * 100

    color_info = []
    for i, color in enumerate(merged_rgb):
        hex_color = get_hex_code(color)
        percentage = percentages[i]
        color_info.append((hex_color, percentage))

    return color_info


def retrieve_dominant_colors_slic(img, n_segments=20, compactness=30):
    """
    This method retrieves dominant colors by treating the image as a 
    collection of superpixels. Images are converted to the CIELAB
    color space (to account for visual similarity between colors) and
    returns closely similar regions (in terms of color), which are selected
    using SLIC (Simple Linear Iterative Clustering).
    """
    float_img = img_as_float(img)
    lab_img = rgb2lab(float_img)
    segments = slic(lab_img, n_segments=n_segments, compactness=compactness, start_label=1)
    color_regions = defaultdict(lambda: { "pixels": 0, "color_sum": np.array([0.0, 0.0, 0.0]) })

    for (y, x), label in np.ndenumerate(segments):
        color = lab_img[y, x]
        color_regions[label]["pixels"] += 1
        color_regions[label]["color_sum"] += color

    total_pixels = img.shape[0] * img.shape[1]
    for label, region in color_regions.items():
        avg_lab_color = region["color_sum"] / region["pixels"]
        avg_rgb_color = lab2rgb(avg_lab_color.reshape(1, 1, 3)).squeeze() * 255
        hex_color = get_hex_code(avg_rgb_color)
        percentage = (region["pixels"] / total_pixels) * 100
        color_regions[label].update({ 
            "average_color": avg_rgb_color, 
            "hex_color": hex_color,
            "percentage": percentage
        })

    color_info = []
    sorted_regions = sorted(color_regions.values(), key=lambda r: -r["percentage"])
    for i, region in enumerate(sorted_regions[:10], 1):
        color_info.append((region["hex_color"], region["percentage"]))

    return color_info


def retrieve_dominant_colors(img, strategy="canny"):
    if strategy == "canny":
        return retrieve_dominant_colors_canny(img)
    if strategy == "slic":
        return retrieve_dominant_colors_slic(img)
    if strategy == "lab":
        return retrieve_dominant_colors_lab(img)

    return


def get_color_luminance(hex_code):
    red, green, blue = hex_to_rgb(hex_code)

    # color brightness formula based from the RGB to YIQ conversion formula.
    # see: https://www.w3.org/TR/AERT/#color-contrast
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    return luminance
