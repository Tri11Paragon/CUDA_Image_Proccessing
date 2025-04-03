import argparse

import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader

import feed_forward as ff
import basic_classify as classify
import camera
import threading

import numpy as np

import time
import sys
import nxt
import nxt.locator
import nxt.motor
import nxt.sensor
import nxt.sensor.generic
import torch.nn as nn

command_lock = threading.Lock()

def execute_command(command_func):
    def command_with_guard():
        # Acquire the lock before executing the command
        with command_lock:
            print(f"Running command {command_func.__name__}")
            command_func()

    if not command_lock.locked():
        # Run the blocking command on a separate thread
        thread = threading.Thread(target=command_with_guard)
        thread.start()


MAX_POINTS = 512
IMAGE_SIZE = 24
CLASSES = 3
NUM_COLORS = 3


class ContourClassifier(nn.Module):
    def __init__(self, input_size, image_size):
        super().__init__()
        print(input_size, image_size)
        self.classifier = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_size, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 128),
                nn.LeakyReLU(),
            ),
            nn.LeakyReLU(),
            nn.Linear(128, 128 * 3),
            nn.LeakyReLU(),
            nn.Linear(128 * 3, 128 * 3),
            nn.LeakyReLU(),
            nn.Linear(128 * 3, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.LeakyReLU(),
            nn.Linear(16, CLASSES),
            # nn.Softmax(1)
        )
        self.color_predictor = nn.Sequential(
            nn.Linear(image_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_COLORS),
            # nn.Softmax(1)
        )

    def forward(self, x):
        # data = self.class_begin(x[0])
        # atten, _ = self.attention(data, data, data)
        return self.classifier(x[0]), self.color_predictor(x[1])


def bumper(sensor):
    def bumpy():
        while not sensor.get_sample():
            pass
        return True

    return bumpy


def prep():
    global brick
    global motor_shoulder
    global motor_elbow
    global touch_shoulder
    global touch_elbow
    try:
        brick = nxt.locator.find()
    except nxt.locator.BrickNotFoundError:
        print("--\n<<< Dud yiu asdjklahd:--")
        sys.exit(0)
        # if sys.flag.interactive:
        #     return
        # else:
        #     sys.exit(0)
    motor_shoulder = brick.get_motor(nxt.motor.Port.A)
    motor_elbow = brick.get_motor(nxt.motor.Port.B)
    touch_shoulder = brick.get_sensor(nxt.sensor.Port.S1, nxt.sensor.generic.Touch)
    touch_elbow = brick.get_sensor(nxt.sensor.Port.S2, nxt.sensor.generic.Touch)


def cleanup():
    motor_shoulder.idle()
    motor_elbow.idle()
    brick.close()


def home():
    motor_shoulder.turn(-15, 360, stop_turn=bumper(touch_shoulder))
    motor_elbow.turn(15, 360, stop_turn=bumper(touch_elbow))


def left():
    home()
    motor_shoulder.turn(30, 200, True)
    motor_elbow.turn(-15, 120, True)
    home()


def right():
    home()
    motor_elbow.turn(-20, 240, True)
    motor_shoulder.turn(20, 120, True)
    motor_elbow.turn(15, 180, True)
    home()


def forward():
    home()
    motor_elbow.turn(-20, 240, True)
    motor_shoulder.turn(20, 100, True)
    motor_elbow.turn(50, 50, True)
    home()


def handle_image(image, model):
    if image is None:
        return
    draw = image.copy()
    classification_data = classify.classify(image)

    contour_data = []
    bounded_images = []
    for best_color, best_shape, bound, c, corners in classification_data:
        x, y, w, h = bound
        contour_features, con_len = ff.preprocess_contour(c)
        contour_data.append(contour_features)
        bounded_images.append(ff.preprocess_bounds(image, x, y, w, h))

    contour_tensor = torch.tensor(np.array(contour_data), dtype=torch.float32)
    image_tensor = torch.tensor(np.array(bounded_images), dtype=torch.float32)

    test_dataset = TensorDataset(contour_tensor, image_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch_contours, batch_images in test_loader:
            class_pred, color_pred = model([batch_contours, batch_images])

            predicted_classes = torch.argmax(class_pred, dim=1)
            predicted_color = torch.argmax(color_pred, dim=1)

            for class_data, color, classification in zip(classification_data, predicted_color, predicted_classes):
                best_color, best_shape, bound, c, corners = class_data
                x, y, w, h = bound
                cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(draw, "Best Shape: " + best_shape, (x - 2, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255),
                            1, cv2.LINE_AA)
                cv2.putText(draw, "Predicted Color (NN): " + ff.get_string_from_color(color),
                            (x - 2, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            1, cv2.LINE_AA)
                cv2.putText(draw, "Predicted Shape (NN): " + ff.get_string_from_class(classification), (x - 2, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255),
                            1, cv2.LINE_AA)
                cv2.putText(draw, "Best Color: " + best_color, (x - 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255),
                            1, cv2.LINE_AA)
                key = cv2.waitKey(1)
                if key == ord('r'):
                    if best_shape.startswith('Z') or best_shape.startswith('L'):
                        if best_shape.startswith('Z') and color == 2:
                            execute_command(left)
                        elif best_shape.startswith('Z') and color == 1:
                            execute_command(right)
                        elif best_shape.startswith('L') and color == 1:
                            execute_command(left)
                        elif best_shape.startswith('L') and color == 2:
                            execute_command(right)
                        else:
                            execute_command(forward)
                    else:
                        execute_command(forward)
    cv2.imshow("Meow", draw)


def main():
    parser = argparse.ArgumentParser(prog="integration")
    parser.add_argument("model")
    args = parser.parse_args()

    model = ff.load_model(args.model)
    model.eval()

    # prep()
    camera.open_camera("local", lambda image, w, h: handle_image(image, model))


if __name__ == "__main__":
    main()
