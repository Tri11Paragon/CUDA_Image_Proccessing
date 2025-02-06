import os
import cv2
import argparse
import numpy as np
import datetime
import camera
import process
import basic_classify as c
from pathlib import Path
import pickle
import sqlite3
import random


def generate_flipped(img):
    img_h = cv2.flip(img, 1)
    img_v = cv2.flip(img, 0)
    img_hv = cv2.flip(img, -1)
    return img_h, img_v, img_hv


def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = np.clip(v + value, 0, 255)
    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REFLECT)

    return rotated


def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy = image.copy()
    total_pixels = image.size // image.shape[-1]  # Number of pixels

    # Salt (white) noise
    num_salt = int(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = [255, 255, 255]

    # Pepper (black) noise
    num_pepper = int(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = [0, 0, 0]

    return noisy


def add_poisson_noise(image):
    noisy = np.random.poisson(image.astype(np.float32))
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def run_generate(args):
    files = list(Path(args.res_folder).iterdir())
    for file in files:
        if file.is_folder():
            files.append()




def main():
    parser = argparse.ArgumentParser(description="Preprocess images for use in the neural network")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate images")
    generate_parser.add_argument('-d', action='store_true', default=False, help="Add images to database of bounds and contours")
    generate_parser.add_argument('-a', action='store_true', default=False, help="Generate all variations of the images (vertical flip, horizontal flip, both)")
    generate_parser.add_argument('-v', action='store_true', default=False, help="Generate vertical flip variations of the images")
    generate_parser.add_argument('-z', action='store_true', default=False, help="Generate horizontal flip variations of the images")
    generate_parser.add_argument('-b', action='store_true', default=False, help="Generate images with both flips applied")
    generate_parser.add_argument('-c', action='store_true', default=False, help="Copy original image as well")
    generate_parser.add_argument('-u', action='store_true', default=False, help="Give images a unique id instead of their existing filename")
    generate_parser.add_argument('-t', action='store', default='jpg', help="Image type to output")
    generate_parser.add_argument("res_folder", help="Folder to take images from, will maintain folder structure")
    generate_parser.add_argument("output_folder", help="Folder to put the generated images into")

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)


if __name__ == "__main__":
    main()
