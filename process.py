import os
import cv2
import argparse
import numpy as np
import datetime

def threshold_image(image):
    size = 15
    kernel = np.ones((size, size), np.float32) / (size * size)

    inverted = cv2.bitwise_not(image)

    blured = cv2.filter2D(inverted, -1, kernel)
    blured = cv2.GaussianBlur(blured, (size, size), 0)

    lower_bound = np.array([150, 0, 0])  # Lower BGR threshold
    upper_bound = np.array([255, 255, 255])  # Upper BGR threshold

    mask = cv2.inRange(blured, lower_bound, upper_bound)
    result = cv2.bitwise_and(blured, blured, mask=mask)

    return result


def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log scale transform for better numerical stability
    hu_moments = -np.sign(hu_moments) * np.log(np.abs(hu_moments) + 1e-7)
    return hu_moments


def find_bounds_and_contours(grey_image, limit=10, min_dist_to_edge=10):
    contours, _ = cv2.findContours(grey_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounds = [(cv2.boundingRect(c), c) for c in contours]

    h, w = grey_image.shape

    return [x for x in bounds
            if x[0][2] > limit and x[0][3] > limit and not (
                x[0][0] < min_dist_to_edge or
                x[0][1] < min_dist_to_edge or
                (x[0][0] + x[0][2] > (w - min_dist_to_edge)) or
                (x[0][1] + x[0][3] > (h - min_dist_to_edge)))]

reference_thresholds = {
    "T-Shape": cv2.cvtColor(threshold_image(cv2.imread("res/grey_t/2025-02-03 19:29:49.845166.png")), cv2.COLOR_BGR2GRAY),
    "Z-Shape": cv2.cvtColor(threshold_image(cv2.imread("res/green_z/2025-02-03 19:32:15.596875.png")), cv2.COLOR_BGR2GRAY),
}

reference_shapes = {
    "T-Shape": (find_bounds_and_contours(reference_thresholds["T-Shape"])[0][1]),
    "Z-Shape": (find_bounds_and_contours(reference_thresholds["Z-Shape"])[0][1])
}


def process_image(image, width, height):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = threshold_image(image)
    thresh_grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    bounds = find_bounds_and_contours(thresh_grey)

    for bound, c in bounds:
        x, y, w, h = bound

        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        num_corners = len(approx)

        best_shape = "Unknown"
        if num_corners == 6:
            best_shape = "L-Shape"
        else:
            min_diff = float("inf")
            # hu_moments = get_hu_moments(c)
            for shape_name, ref in reference_shapes.items():
                score = cv2.matchShapes(c, ref, cv2.CONTOURS_MATCH_I1, 0)
                if score < min_diff:
                    min_diff = score
                    best_shape = shape_name
                # distance = np.linalg.norm(hu_moments - ref_hu)
                # if distance < min_diff:
                #     min_diff = distance
                #     best_shape = shape_name

        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        solidity = float(area) / hull_area if hull_area > 0 else 0

        cv2.drawContours(thresh, [c], -1, (255, 0, 0), 2)
        cv2.drawContours(thresh, [hull], -1, (0, 0, 255), 2)

        cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(thresh, "Corners: " + str(num_corners), (x - 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    1, cv2.LINE_AA)
        cv2.putText(thresh, best_shape, (x - 2, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # combined = cv2.addWeighted(edges, 0.5, thresh, 0.5, 0)

    cv2.imshow("edges", thresh)


def open_camera(save_path, frame_call):
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        try:
            ret, frame = cam.read()

            frame_call(frame, frame_width, frame_height)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if save_path and key == ord('s'):
                now = datetime.datetime.now()
                if not save_path.endswith('/'):
                    save_path += '/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path + '/' + str(now) + '.png', frame)
        except KeyboardInterrupt:
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action='store_true', default=False, help="Use webcam as input instead of images")
    parser.add_argument('-a', action='store_true', default=False,
                        help="Use an average of the input on the webcam instead of the current image frame")
    parser.add_argument('-i', action='store', help="Image path to use if you are not using a webcam")
    parser.add_argument('-s', action='store', help="Path to save images in")

    args = parser.parse_args()

    if not args.i and not args.w:
        print("Please select a mode of operation!")
        print("--help to see a list of available commands")
        print("\t-w\t\tUse your webcam as input")
        print("\t-i\t\tUse a static image as input")

    if args.i:
        img = cv2.imread(args.i)
        width, height, channels = img.shape
        while True:
            process_image(img, width, height)

            if cv2.waitKey(1) == ord('q'):
                break

    if args.w:
        open_camera(args.s, process_image)


if __name__ == '__main__':
    main()
