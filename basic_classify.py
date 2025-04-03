import process
import cv2
import numpy as np

reference_images = {
    "T1": cv2.imread(
        "res/base/green_t/2025-02-05 13:54:52.266144.png"),
    "Z1": cv2.imread(
        "res/base/green_z/2025-02-03 19:32:15.596875.png"),
    "L1": cv2.imread(
        "res/base/green_l/2025-02-03 19:33:51.765660.png"),
    "T2": cv2.imread("res/base/grey_t/2025-02-03 19:29:32.805979.png"),
    "Z2": cv2.imread("res/base/grey_z/2025-02-03 19:24:16.587044.png"),
    "L2": cv2.imread("res/base/grey_l/2025-02-03 19:27:52.109698.png"),
    "T3": cv2.imread(
        "res/base/orange_t/2025-02-03 19:38:09.115254.png"),
    "Z3": cv2.imread(
        "res/base/orange_z/2025-02-03 19:22:16.808573.png"),
    "L3": cv2.imread(
        "res/base/orange_l/2025-02-03 19:36:11.811343.png")
}

reference_thresholds = {
    "T-Shape-Green": cv2.cvtColor(process.threshold_image(reference_images["T1"]),
                                  cv2.COLOR_BGR2GRAY),
    "Z-Shape-Green": cv2.cvtColor(process.threshold_image(reference_images["Z1"]),
                                  cv2.COLOR_BGR2GRAY),
    "L-Shape-Green": cv2.cvtColor(process.threshold_image(reference_images["L1"]),
                                  cv2.COLOR_BGR2GRAY),
    "T-Shape-Grey": cv2.cvtColor(process.threshold_image(reference_images["T2"]),
                                 cv2.COLOR_BGR2GRAY),
    "Z-Shape-Grey": cv2.cvtColor(process.threshold_image(reference_images["Z2"]),
                                 cv2.COLOR_BGR2GRAY),
    "L-Shape-Grey": cv2.cvtColor(process.threshold_image(reference_images["L2"]),
                                 cv2.COLOR_BGR2GRAY),
    "T-Shape-Orange": cv2.cvtColor(process.threshold_image(reference_images["T3"]),
                                   cv2.COLOR_BGR2GRAY),
    "Z-Shape-Orange": cv2.cvtColor(process.threshold_image(reference_images["Z3"]),
                                   cv2.COLOR_BGR2GRAY),
    "L-Shape-Orange": cv2.cvtColor(process.threshold_image(reference_images["L3"]),
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

h_bins = 120
s_bins = 120
v_bins = 120
histSize = [h_bins, s_bins, v_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
v_ranges = [0, 256]
ranges = h_ranges + s_ranges + v_ranges
channels = [0, 1, 2]

def classify(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    thresh = process.threshold_image(image)
    thresh_grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    bounds = process.find_bounds_and_contours(thresh_grey)

    data = []
    for bound, c in bounds:
        epsilon = 0.03 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        num_corners = len(approx)

        image_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(image_mask, [c], -1, 255, -1)
        mean_color_c1 = cv2.mean(image, mask=image_mask)
        hist_image = cv2.calcHist([hsv_image], channels, image_mask, histSize, ranges)
        cv2.normalize(hist_image, hist_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        best_shape = "U0"
        best_color = "Orange"
        best_correl_color = "Orange"
        # if num_corners == 6:
        #     best_shape = "L0"
        # else:
        min_diff = float("inf")
        min_color = float("inf")
        best_correl = 0
        for shape_name, ref in reference_shapes.items():
            contour_mask = np.zeros(reference_images[shape_name].shape[:2], dtype=np.uint8)
            hsv_image_shape = cv2.cvtColor(reference_images[shape_name], cv2.COLOR_BGR2HSV)
            cv2.drawContours(contour_mask, [ref], -1, 255, -1)

            mean_color_c2 = cv2.mean(reference_images[shape_name], mask=contour_mask)
            hist_c2 = cv2.calcHist([hsv_image_shape], channels, contour_mask, histSize, ranges)
            cv2.normalize(hist_c2, hist_c2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            comparison = cv2.compareHist(hist_image, hist_c2, cv2.HISTCMP_CORREL)

            distance = np.linalg.norm(np.array(mean_color_c1[:3]) - np.array(mean_color_c2[:3]))

            if shape_name.endswith("1"):
                our_color = "Green"
            elif shape_name.endswith("2"):
                our_color = "Grey"
            else:
                our_color = "Orange"

            score = cv2.matchShapes(c, ref, cv2.CONTOURS_MATCH_I1, 0)
            if score < min_diff:
                min_diff = score
                best_shape = shape_name
            if distance < min_color:
                min_color = distance
                best_color = our_color
            if comparison > best_correl:
                best_correl = comparison
                best_correl_color = our_color

        data.append((best_correl_color, best_shape, bound, c, num_corners))

    return data

def classify_color(color):
    if color[2] > 150:
        return "orange"
    elif color[0] > 70 and color[1] > 70 and color[2] >= 60:
        return "grey"
    return "green"

def draw_classification(image):
    draw = image.copy()
    data = classify(image)

    for shape, bound, c, corners in data:
        x, y, w, h = bound

        hull = cv2.convexHull(c)

        cv2.drawContours(draw, [c], -1, (255, 0, 0), 2)
        cv2.drawContours(draw, [hull], -1, (0, 0, 255), 2)

        average_color = process.extract_color(image, c)
        p_color = classify_color(average_color)

        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(draw, "Corners: " + str(corners), (x - 2, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(draw, str(round(average_color[0], 2)) + " " + str(round(average_color[1], 2)), (x - 2, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(draw, str(round(average_color[2], 2)), (x - 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(draw, shape + " " + p_color, (x - 2, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Classification", draw)