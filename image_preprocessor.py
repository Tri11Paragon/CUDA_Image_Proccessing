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


def image_to_database(image, path, original, cursor):
    thresh = process.threshold_image(image)
    thresh_grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    bounds = process.find_bounds_and_contours(thresh_grey)

    cursor.execute("""
                INSERT INTO images (filename, original_file)
                VALUES (?, ?)
            """, (path, original))

    image_id = cursor.lastrowid
    print(image_id)

    for bound, con in bounds:
        x, y, w, h = bound
        serialized_contour = pickle.dumps(con)

        cursor.execute("""
                INSERT INTO contours (image_id, contour)
                VALUES (?, ?)
            """, (image_id, serialized_contour))

        cursor.execute("""
                INSERT INTO bounds (image_id, x, y, width, height)
                VALUES (?, ?, ?, ?, ?)
            """, (image_id, x, y, w, h))



def run_generate(args):
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()

    if args.d:
        connection = sqlite3.connect(Path(args.output_folder).joinpath("image_data.db"))
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                original_file TEXT)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contours (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                contour BLOB,
                FOREIGN KEY(image_id) REFERENCES images(id)
            )
        """)
        connection.commit()

    files = list(Path(args.res_folder).iterdir())
    i = 0
    file_name = 0
    while i < len(files):
        file = files[i]
        i += 1
        if file.is_dir():
            files.extend(list(Path(file.resolve()).iterdir()))
        else:
            image = cv2.imread(str(file.resolve()))
            h, v, b = generate_flipped(image)

            new_image_name = Path(args.output_folder).joinpath(file.with_suffix('.' + args.t).name)

            if args.u:
                new_image_name = new_image_name.with_stem(str(file_name))
                file_name += 1

            if args.i:
                new_image_name = new_image_name.with_stem(new_image_name.stem + "_" + file.parents[0].name)

            if args.c:
                cv2.imwrite(str(new_image_name.resolve()), image)
                if args.d:
                    image_to_database(image, str(new_image_name.resolve().relative_to(cwd)), str(file.resolve().relative_to(cwd)), cursor)
                    connection.commit()

            if args.a or args.z:
                h_path = new_image_name.with_stem(new_image_name.stem + "_hflip").resolve()
                cv2.imwrite(str(h_path), h)
                if args.d:
                    image_to_database(h, str(h_path.relative_to(cwd)), str(file.resolve().relative_to(cwd)), cursor)
                    connection.commit()
            if args.a or args.v:
                v_path = new_image_name.with_stem(new_image_name.stem + "_vflip").resolve()
                cv2.imwrite(str(v_path), v)
                if args.d:
                    image_to_database(v, str(v_path.relative_to(cwd)), str(file.resolve().relative_to(cwd)), cursor)
                    connection.commit()
            if args.a or args.b:
                b_path = new_image_name.with_stem(new_image_name.stem + "_bflip").resolve()
                cv2.imwrite(str(b_path), b)
                if args.d:
                    image_to_database(b, str(b_path.relative_to(cwd)), str(file.resolve().relative_to(cwd)), cursor)
                    connection.commit()

            print(f"Processed {new_image_name}")

    if args.d:
        connection.close()



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
    generate_parser.add_argument('-i', action='store_true', default=False, help="Insert the last directory name into the name of the image. Good for use with -u")
    generate_parser.add_argument('-t', action='store', default='jpg', help="Image type to output")
    generate_parser.add_argument("res_folder", help="Folder to take images from")
    generate_parser.add_argument("output_folder", help="Folder to put the generated images into")

    args = parser.parse_args()

    if args.mode == "generate":
        run_generate(args)


if __name__ == "__main__":
    main()
