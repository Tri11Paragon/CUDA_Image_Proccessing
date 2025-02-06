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

def handle_test(bool1, bool2, name, correct_shape_color, correct_shape, correct_color, incorrect):
    if bool1:
        if bool2:
            correct_shape_color.append(name)
        else:
            correct_shape.append(name)
    else:
        if bool2:
            correct_color.append(name)
        else:
            incorrect.append(name)


def run_test(folder):
    correct_shape_color = []
    correct_shape = []
    correct_color = []
    incorrect = []
    too_many = []
    no_classification = []
    for file in Path(folder).iterdir():
        if not file.is_file():
            continue
        print(f"Processing {folder}/{file.name}")

        image = cv2.imread(str(file.resolve()))

        data = c.classify(image)

        if len(data) == 0:
            no_classification.append(file.name)
        elif len(data) > 1:
            too_many.append(file.name)
        else:
            shape, bound, con, corners = data[0]

            color = process.extract_color(image, con)
            p_color = c.classify_color(color)

            name_split = folder.split('_')
            c_color = name_split[0].split('/')[-1]
            c_shape = name_split[1]

            if c_shape == 'l':
                handle_test(shape.startswith('L'), p_color == c_color, file.name, correct_shape_color, correct_shape, correct_color, incorrect)
            elif c_shape == 't':
                handle_test(shape.startswith('T'), p_color == c_color, file.name, correct_shape_color, correct_shape, correct_color, incorrect)
            elif c_shape == 'z':
                handle_test(shape.startswith('Z'), p_color == c_color, file.name, correct_shape_color, correct_shape, correct_color, incorrect)
            else:
                print(f"unable to classify object path? {folder}")
    return folder, correct_shape_color, correct_shape, correct_color, incorrect, too_many, no_classification

def run_tests():
    results = []
    if Path("results.pkl").exists():
        with open("results.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results.append(run_test("res/green_l"))
        results.append(run_test("res/green_t"))
        results.append(run_test("res/green_z"))
        results.append(run_test("res/grey_l"))
        results.append(run_test("res/grey_t"))
        results.append(run_test("res/grey_z"))
        results.append(run_test("res/orange_l"))
        results.append(run_test("res/orange_t"))
        results.append(run_test("res/orange_z"))

        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)

    total_correct = 0
    total_incorrect = 0

    for folder, correct_shape_color, correct_shape, correct_color, incorrect, too_many, no_classification in results:
        for path in correct_color:
            print(f"Color: {folder}/{path}")
        for path in correct_shape:
                print(f"Shape: {folder}/{path}")

    for folder, correct_shape_color, correct_shape, correct_color, incorrect, too_many, no_classification in results:
        total_correct += len(correct_shape_color)
        total_partially_correct = len(correct_shape) + len(correct_color)
        total_incorrect += len(incorrect) + total_partially_correct

        print(f"For folder {folder}:")
        print(f"\tCorrect Color & Shape: {len(correct_shape_color)}")
        print(f"\tCorrect Color: {len(correct_color)}")
        print(f"\tCorrect Shape: {len(correct_shape)}")
        print()
        print(f"\tIncorrect: {len(incorrect)}")
        print(f"\tToo many: {len(too_many)}")
        print(f"\tNo classification: {len(no_classification)}")

    print(f"Total correct: {total_correct}")
    print(f"Total incorrect: {total_incorrect}")
    print(f"Accuracy: {(total_correct / (total_incorrect + total_correct)) * 100}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', action='store_true', default=False, help="Use webcam as input instead of images")
    parser.add_argument('-a', action='store_true', default=False,
                        help="Use an average of the input on the webcam instead of the current image frame")
    parser.add_argument('-i', action='store', help="Image path to use if you are not using a webcam")
    parser.add_argument('-s', action='store', help="Path to save images in")
    parser.add_argument('-t', action='store_true', help="Test images in res")

    args = parser.parse_args()

    if not args.i and not args.w and not args.t:
        print("Please select a mode of operation!")
        print("--help to see a list of available commands")

    if args.i:
        img = cv2.imread(args.i)
        width, height, channels = img.shape
        while True:
            c.draw_classification(img)

            if cv2.waitKey(1) == ord('q'):
                break

    if args.w:
        camera.open_camera(args.s, c.draw_classification)

    if args.t:
        run_tests()

if __name__ == '__main__':
    main()