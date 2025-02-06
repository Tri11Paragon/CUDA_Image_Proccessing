import process
import cv2
import numpy as np

reference_thresholds = {
    "T-Shape-Green": cv2.cvtColor(process.threshold_image(cv2.imread("res/green_t/2025-02-05 13:54:52.266144.png")),
                                  cv2.COLOR_BGR2GRAY),
    "Z-Shape-Green": cv2.cvtColor(process.threshold_image(cv2.imread("res/green_z/2025-02-03 19:32:15.596875.png")),
                                  cv2.COLOR_BGR2GRAY),
    "L-Shape-Green": cv2.cvtColor(process.threshold_image(cv2.imread("res/green_l/2025-02-03 19:33:51.765660.png")),
                                  cv2.COLOR_BGR2GRAY),
    "T-Shape-Grey": cv2.cvtColor(process.threshold_image(cv2.imread("res/grey_t/2025-02-03 19:29:32.805979.png")),
                                 cv2.COLOR_BGR2GRAY),
    "Z-Shape-Grey": cv2.cvtColor(process.threshold_image(cv2.imread("res/grey_z/2025-02-03 19:24:16.587044.png")),
                                 cv2.COLOR_BGR2GRAY),
    "L-Shape-Grey": cv2.cvtColor(process.threshold_image(cv2.imread("res/grey_l/2025-02-03 19:27:52.109698.png")),
                                 cv2.COLOR_BGR2GRAY),
    "T-Shape-Orange": cv2.cvtColor(process.threshold_image(cv2.imread("res/orange_t/2025-02-03 19:38:09.115254.png")),
                                   cv2.COLOR_BGR2GRAY),
    "Z-Shape-Orange": cv2.cvtColor(process.threshold_image(cv2.imread("res/orange_z/2025-02-03 19:22:16.808573.png")),
                                   cv2.COLOR_BGR2GRAY),
    "L-Shape-Orange": cv2.cvtColor(process.threshold_image(cv2.imread("res/orange_l/2025-02-03 19:36:11.811343.png")),
                                   cv2.COLOR_BGR2GRAY)
}

reference_shapes = {
    "T1": (process.find_bounds_and_contours(reference_thresholds["T-Shape-Green"])[0][1]),
    "Z1": (process.find_bounds_and_contours(reference_thresholds["Z-Shape-Green"])[0][1]),
    "L1": (process.find_bounds_and_contours(reference_thresholds["L-Shape-Green"])[0][1]),
    "T2": (process.find_bounds_and_contours(reference_thresholds["T-Shape-Grey"])[0][1]),
    "Z2": (process.find_bounds_and_contours(reference_thresholds["Z-Shape-Grey"])[0][1]),
    "L2": (process.find_bounds_and_contours(reference_thresholds["L-Shape-Grey"])[0][1]),
    "T3": (process.find_bounds_and_contours(reference_thresholds["T-Shape-Orange"])[0][1]),
    "Z3": (process.find_bounds_and_contours(reference_thresholds["Z-Shape-Orange"])[0][1]),
    "L3": (process.find_bounds_and_contours(reference_thresholds["L-Shape-Orange"])[0][1])
}

def classify(image):
    thresh = process.threshold_image(image)
    thresh_grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    bounds = process.find_bounds_and_contours(thresh_grey)

    data = []
    for bound, c in bounds:
        epsilon = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        num_corners = len(approx)

        best_shape = "U0"
        if num_corners == 6:
            best_shape = "L0"
        else:
            min_diff = float("inf")
            for shape_name, ref in reference_shapes.items():
                score = cv2.matchShapes(c, ref, cv2.CONTOURS_MATCH_I1, 0)
                if score < min_diff:
                    min_diff = score
                    best_shape = shape_name

        data.append((best_shape, bound, c, num_corners))

    return data

def draw_classification(image):
    draw = image.copy()
    data = classify(image)

    for shape, bound, c, corners in data:
        x, y, w, h = bound

        hull = cv2.convexHull(c)

        cv2.drawContours(draw, [c], -1, (255, 0, 0), 2)
        cv2.drawContours(draw, [hull], -1, (0, 0, 255), 2)

        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(draw, "Corners: " + str(corners), (x - 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(draw, shape, (x - 2, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Classification", draw)