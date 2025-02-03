import cv2
import argparse
import numpy as np

image_average = []

def process_image(image, width, height):
    # global image_average
    # image_average =
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    #
    # x = cv2.convertScaleAbs(x)
    # y = cv2.convertScaleAbs(y)
    #
    # edges = cv2.addWeighted(x, 0.5, y, 0.5, 0)

    edges = cv2.Canny(image, 50, 100)
    cv2.imshow("edges", edges)

def open_camera(frame_call):
    cam = cv2.VideoCapture(0)
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        try:
            ret, frame = cam.read()

            frame_call(frame, frame_width, frame_height)

            if cv2.waitKey(1) == ord('q'):
                break
        except KeyboardInterrupt:
            break
    cam.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action='store_true', default=False, help="Use webcam as input instead of images")
    parser.add_argument('-a', action='store_true', default=False, help="Use an average of the input on the webcam instead of the current image frame")
    parser.add_argument('-i', action='store', help="Image path to use if you are not using a webcam")

    args = parser.parse_args()

    if not args.i and not args.w:
        print("Please select a mode of operation!")
        print("--help to see a list of available commands")
        print("\t-w\t\tUse your webcam as input")
        print("\t-i\t\tUse a static image as input")

    if args.i:
        img = cv2.imread(args.i)
        height, width, channels = img.shape
        while True:
            process_image(img, width, height)

            if cv2.waitKey(1) == ord('q'):
                break

    if args.w:
        open_camera(process_image)

if __name__ == '__main__':
    main()