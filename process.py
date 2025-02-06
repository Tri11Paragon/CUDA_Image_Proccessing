import os
import cv2
import argparse
import numpy as np
import datetime
import camera


def modify_hsv(image, func):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h, s, v = func(h, s, v)
    hsv_enhanced = cv2.merge([np.clip(h, 0, 255).astype(np.uint8),
                              np.clip(s, 0, 255).astype(np.uint8),
                              np.clip(v, 0, 255).astype(np.uint8)])

    return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)


def sharpen(image, intensity=5):
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, intensity, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, sharpening_kernel)


def threshold_image(image):
    size = 5
    kernel = np.ones((size, size), np.uint8)

    saturated = image.copy()
    saturated = modify_hsv(saturated, lambda h, s, v: (h, s, v * 1.9))
    saturated = modify_hsv(saturated, lambda h, s, v: (h, s, v / 1.9))
    saturated = modify_hsv(saturated, lambda h, s, v: ((h + 30) % 180, s, v))
    saturated = modify_hsv(saturated, lambda h, s, v: ((h + 90) % 180, s * 0.5, v * 2.5))

    gray = cv2.cvtColor(saturated, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 100)
    edges = cv2.dilate(edges, kernel)
    colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return colored


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


def extract_color(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_color = cv2.mean(image, mask=mask)
    return mean_color[:3]


def draw_bounds(image, limit=10, min_dist_to_edge=10):
    local = image.copy()
    contours = find_bounds_and_contours(cv2.cvtColor(local, cv2.COLOR_BGR2GRAY), limit, min_dist_to_edge)

    for bound, c in contours:
        x, y, w, h = bound
        cv2.rectangle(local, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return local


def draw_contours(image, limit=10, min_dist_to_edge=10):
    mask = np.zeros_like(image)
    contours = find_bounds_and_contours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), limit, min_dist_to_edge)

    for bound, c in contours:
        cv2.drawContours(mask, [c], -1, (255, 255, 255), 2)

    return mask


def main():
    parser = argparse.ArgumentParser(description='Capture video and display threshold values')

    parser.add_argument('-i', action='store', help='image path to use')
    parser.add_argument('-w', action='store', nargs='?', help='use webcam flag', default=None, const=0, type=int)

    args = parser.parse_args()

    if args.w:
        camera.open_camera("unused",
                           lambda image, w, h: cv2.imshow("Camera View", cv2.hconcat([threshold_image(image), image])))
    elif args.i:
        img = cv2.imread(args.i)
        while True:
            thresh = threshold_image(img)
            p1 = cv2.vconcat([thresh, draw_bounds(thresh)])
            p2 = cv2.vconcat([img, draw_contours(thresh)])
            cv2.imshow("Image View", cv2.hconcat([p1, p2]))

            if cv2.waitKey(1) == ord('q'):
                break
    else:
        print("Please provide either -i or -w flags")


if __name__ == '__main__':
    main()
