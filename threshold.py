import os
import cv2
import argparse
import numpy as np
import datetime
import process

def main():
    parser = argparse.ArgumentParser(description='Capture video and display threshold values')

    parser.add_argument('-i', action='store', help='image path to use')
    parser.add_argument('-w', action='store', nargs='?', help='use webcam flag', const=0, type=int)

    args = parser.parse_args()


    process.open_camera("unused", lambda image, w, h : cv2.imshow("Camera View", process.threshold_image(image)))

if __name__ == '__main__':
    main()
