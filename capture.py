import os
import cv2
import argparse
import numpy as np
import datetime
import process

def main():
    parser = argparse.ArgumentParser(description='Capture video')
    parser.add_argument("path")

    args = parser.parse_args()

    process.open_camera(args.path, lambda image, w, h : cv2.imshow("Camera View", image))

if __name__ == '__main__':
    main()
