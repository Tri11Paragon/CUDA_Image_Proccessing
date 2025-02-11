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
import shutil


def generate_flipped(img):
    img_h = cv2.flip(img, 1)
    img_v = cv2.flip(img, 0)
    img_hv = cv2.flip(img, -1)
    return img_h, img_v, img_hv


def add_gaussian_noise(image, factor=0.5, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(image, 1 - factor, noise, factor, 0)
    return noisy_image


def adjust_brightness(image, value=0.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = np.clip(v * value, 0, 255).astype(np.uint8)
    hsv_c = cv2.merge([np.clip(h, 0, 255).astype(np.uint8), np.clip(s, 0, 255).astype(np.uint8), v])

    return cv2.cvtColor(hsv_c, cv2.COLOR_HSV2BGR)


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


def add_poisson_noise(image, factor=1):
    noisy = image
    for i in range(0, int(factor)):
        noisy = np.random.poisson(noisy.astype(np.float32))
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

        cursor.execute("""
                        CREATE TABLE IF NOT EXISTS extra_images (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            image_id INTEGER,
                            filename TEXT,
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
            if args.d:
                cursor.execute("SELECT COUNT(*) FROM images WHERE original_file = ?",
                               (str(file.resolve().relative_to(cwd)),))
                if cursor.fetchone()[0] > 0:
                    continue

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
                    image_to_database(image, str(new_image_name.resolve().relative_to(cwd)),
                                      str(file.resolve().relative_to(cwd)), cursor)
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


def extra_image_to_database(path, original_id, cursor):
    cursor.execute("""
                    INSERT INTO extra_images (image_id, filename)
                    VALUES (?, ?)
                """, (original_id, path))


def get_all_images(cursor):
    cursor.execute("SELECT * FROM images")
    return cursor.fetchall()


def update_image_contours_and_bounds(image, cursor, image_id):
    new_bounds = process.find_bounds_and_contours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    cursor.execute("DELETE FROM bounds WHERE image_id = ?", (image_id,))
    cursor.execute("DELETE FROM contours WHERE image_id = ?", (image_id, ))

    for bound, con in new_bounds:
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

def update_all_images(args):
    connection = sqlite3.connect(args.database)
    cursor = connection.cursor()

    for image in get_all_images(cursor):
        image_id, filename, original_file = image
        image = cv2.imread(filename)
        print (f"Processing file {filename}")
        update_image_contours_and_bounds(image, cursor, image_id)

def update_image_filename_with_path(cursor, image_id):
    cursor.execute("SELECT filename FROM images WHERE id = ?", (image_id,))
    result = cursor.fetchone()

    if result is None:
        print(f"No image found with ID {image_id}")
        return

    old_filename = result[0]
    old_path = Path(old_filename)

    new_filename = str(old_path.parent / "images" / old_path.name)

    cursor.execute("""
        UPDATE images
        SET filename = ?
        WHERE id = ?
    """, (new_filename, image_id))

    print(f"Updated image ID {image_id}: {old_filename} -> {new_filename}")


def deep_generate(args):
    output_path = Path(args.output_folder)
    image_path = output_path.joinpath("images")
    gaussian = image_path.joinpath("gaussian_noise")
    gaussian.mkdir(parents=True, exist_ok=True)
    salt_pepper = image_path.joinpath("salt_pepper_noise")
    salt_pepper.mkdir(parents=True, exist_ok=True)
    poisson = image_path.joinpath("poisson_noise")
    poisson.mkdir(parents=True, exist_ok=True)
    brightness = image_path.joinpath("brightness")
    brightness.mkdir(parents=True, exist_ok=True)
    database_path = output_path.joinpath("image_data.db")
    if not database_path.exists():
        shutil.copy(args.database, str(database_path.resolve()))

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    for image in get_all_images(cursor):
        image_id, filename, original_file = image
        print(f"ID: {image_id}, Filename: {filename}, Original File: {original_file}")

        image = cv2.imread(str(filename))

        value_min = 0.2
        value_max = 0.9
        value_step = (value_max - value_min) / (args.n - 1)

        noise_min = 0.05
        noise_max = 0.6
        noise_step = (noise_max - noise_min) / (args.n - 1)

        sp_min = 0.02
        sp_max = 0.06
        sp_step = (sp_max - sp_min) / (args.n - 1)

        poisson_noise_image = add_poisson_noise(image, 1)
        path = str(poisson / (Path(filename).stem + "_poisson" + str(1) + Path(filename).suffix))
        cv2.imwrite(str(path), poisson_noise_image)
        extra_image_to_database(path, image_id, cursor)

        for i in range(0, args.n):
            sp_value = sp_min + sp_step * i
            salt_pepper_noise_image = add_salt_and_pepper_noise(image, sp_value, sp_value)
            path = str(salt_pepper / (Path(filename).stem + "_sp" + str(round(sp_value, 5)) + Path(filename).suffix))
            cv2.imwrite(str(path), salt_pepper_noise_image)
            extra_image_to_database(path, image_id, cursor)

        for i in range(0, args.n):
            noise_value = noise_min + noise_step * i
            gaussian_noise_image = add_gaussian_noise(image, noise_value)
            path = str(gaussian / (Path(filename).stem + "_noise" + str(round(noise_value, 5)) + Path(filename).suffix))
            cv2.imwrite(str(path), gaussian_noise_image)
            extra_image_to_database(path, image_id, cursor)

        for i in range(0, args.n):
            brightness_value = value_min + value_step * i
            brightness_image = adjust_brightness(image, brightness_value)
            path = str(brightness / (
                        Path(filename).stem + "_bright" + str(round(brightness_value, 5)) + Path(filename).suffix))
            cv2.imwrite(path, brightness_image)
            extra_image_to_database(path, image_id, cursor)

    connection.commit()
    connection.close()

def build_database(args):
    resource_dir = Path(args.resource_directory)
    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()

    files = list(resource_dir.iterdir())
    i = 0
    insert_data = []
    while i < len(files):
        file = files[i]
        i += 1
        if file.is_dir():
            files.extend(list(Path(file.resolve()).iterdir()))
            continue
        insert_data.append(())

    connection.commit()
    connection.close()


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for use in the neural network")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate images")
    generate_parser.add_argument('-d', action='store_true', default=False,
                                 help="Add images to database of bounds and contours")
    generate_parser.add_argument('-a', action='store_true', default=False,
                                 help="Generate all variations of the images (vertical flip, horizontal flip, both)")
    generate_parser.add_argument('-v', action='store_true', default=False,
                                 help="Generate vertical flip variations of the images")
    generate_parser.add_argument('-z', action='store_true', default=False,
                                 help="Generate horizontal flip variations of the images")
    generate_parser.add_argument('-b', action='store_true', default=False,
                                 help="Generate images with both flips applied")
    generate_parser.add_argument('-c', action='store_true', default=False, help="Copy original image as well")
    generate_parser.add_argument('-u', action='store_true', default=False,
                                 help="Give images a unique id instead of their existing filename")
    generate_parser.add_argument('-i', action='store_true', default=False,
                                 help="Insert the last directory name into the name of the image. Good for use with -u")
    generate_parser.add_argument('-t', action='store', default='jpg', help="Image type to output")
    generate_parser.add_argument("res_folder", help="Folder to take images from")
    generate_parser.add_argument("output_folder", help="Folder to put the generated images into")

    deep_parser = subparsers.add_parser("deep_generate",
                                        help="Generate images with various noise and brightnesses for use in deep learning")
    deep_parser.add_argument("-n", default=10, type=int, help="Number of images to generate")
    deep_parser.add_argument("res_folder", help="Folder to take images from")
    deep_parser.add_argument("output_folder", help="Folder to put the generated images into")
    deep_parser.add_argument("database", help="Database with image information in it (this will be copied)")

    update_parser = subparsers.add_parser("update", help="Update images within the database")
    update_parser.add_argument("database", help="Database with image information in it (this will be copied)")

    build_parser = subparsers.add_parser("build", help="Build an image database from a folder")
    build_parser.add_argument("resource_directory")
    build_parser.add_argument("database_path")

    args = parser.parse_args()

    if args.mode == "generate":
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        run_generate(args)
    if args.mode == "deep_generate":
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)
        deep_generate(args)
    if args.mode == "update":
        update_all_images(args)
    if args.mode == "build":
        build_database(args)

if __name__ == "__main__":
    main()
