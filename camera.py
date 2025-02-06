import os
import cv2
import argparse
import numpy as np
import datetime

def open_camera(save_path, frame_call, camera = 0):
    cam = cv2.VideoCapture(camera)
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
    parser = argparse.ArgumentParser(description='Capture video')
    parser.add_argument("path")

    args = parser.parse_args()

    open_camera(args.path, lambda image, w, h : cv2.imshow("Camera View", image))

if __name__ == '__main__':
    main()
