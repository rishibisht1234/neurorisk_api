import cv2
import numpy as np

def extract_handwriting_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_lengths = [cv2.arcLength(cnt, True) for cnt in contours]
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]

    return np.array([
        np.mean(contour_lengths) if contour_lengths else 0.0,
        np.std(contour_lengths) if contour_lengths else 0.0,
        np.mean(contour_areas) if contour_areas else 0.0,
        np.std(contour_areas) if contour_areas else 0.0,
    ])
