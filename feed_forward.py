import math
import os
import cv2
import argparse
import numpy as np
import datetime
import camera
import process
import basic_classify as c
import image_preprocessor as ip
from pathlib import Path
import pickle
import sqlite3
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

MAX_POINTS = 512
IMAGE_SIZE = 24
CLASSES = 3
NUM_COLORS = 3

TOTAL_IMAGE_SIZE = IMAGE_SIZE * IMAGE_SIZE

def preprocess_contour(contour):
    contour = np.array(contour, dtype=np.float32)
    contour = contour.flatten()

    # Normalize to [0,1]
    contour -= contour.min(axis=0)
    contour /= (contour.max(axis=0) + 1e-6)

    num_points = len(contour)

    if num_points > MAX_POINTS:
        indices = np.linspace(0, num_points - 1, MAX_POINTS * 2, dtype=int)
        contour = contour[indices]

    padded = np.zeros(MAX_POINTS * 2, dtype=np.float32)
    padded[:len(contour)] = contour[:len(contour)]

    return padded, num_points


def preprocess_bounds(image, x, y, w, h):
    cx, cy = x + w // 2, y + h // 2

    x1 = max(cx - math.floor(IMAGE_SIZE / 2), 0)
    y1 = max(cy - math.floor(IMAGE_SIZE / 2), 0)
    x2 = min(cx + math.ceil(IMAGE_SIZE / 2), image.shape[1])
    y2 = min(cy + math.ceil(IMAGE_SIZE / 2), image.shape[0])

    cropped_region = image[y1:y2, x1:x2]

    if cropped_region.shape[0] != IMAGE_SIZE or cropped_region.shape[1] != IMAGE_SIZE:
        cropped_region = cv2.copyMakeBorder(
            cropped_region,
            top=max(0, IMAGE_SIZE - cropped_region.shape[0]),
            bottom=0,
            left=max(0, IMAGE_SIZE - cropped_region.shape[1]),
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
        )

    return np.array(cropped_region, dtype=np.float32).flatten() / 255.0

def get_shape_class_from_filename(path):
    parts = path.split('_')
    t = parts[2]
    if t == 't':
        return 0
    elif t == 'l':
        return 1
    elif t == 'z':
        return 2
    return 0

def get_color_class_from_filename(path):
    parts = path.split('_')
    color = parts[1]
    if color == "grey":
        return 0
    elif color == "green":
        return 1
    elif color == "orange":
        return 2
    return 0

def load_data_from_db(args):
    print("Loading data from database...")
    conn = sqlite3.connect(args.database)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT c.contour, i.filename, b.x, b.y, b.width, b.height FROM contours c INNER JOIN images i ON c.image_id = i.id INNER JOIN bounds b on i.id = b.image_id")
    data = cursor.fetchall()
    conn.close()

    shape_data, color_data, shape_classification, color_classification = [], [], [], []
    contour_length = 0

    for contour_blob, image_path, x, y, width, height in data:
        contour = pickle.loads(contour_blob)
        image = cv2.imread(image_path)

        contour_features, con_len = preprocess_contour(contour)
        color_features = preprocess_bounds(image, x, y, width, height)

        contour_length += con_len

        shape_data.append(contour_features)
        color_data.append(color_features)
        shape_classification.append(get_shape_class_from_filename(image_path))
        color_classification.append(get_color_class_from_filename(image_path))

    print(f"Average contour input length: {contour_length / float(len(data))}")

    return (torch.tensor(np.array(shape_data), dtype=torch.float32), torch.tensor(np.array(color_data), dtype=torch.float32),
            torch.tensor(np.array(shape_classification), dtype=torch.long), torch.tensor(np.array(color_classification), dtype=torch.long))

def load_from_file(file):
    return pickle.loads(open(file, "rb").read())

def save_to_file(path : str, file):
    with open(path, "wb") as f:
        pickle.dump(file, f)

def load_data(args):
    if Path(args.t).exists():
        print("Loading image data from cache file...")
        return load_from_file(args.t)
    return load_data_from_db(args)

def save_data(args, data):
    save_to_file(args.t, data)

class ContourClassifier(nn.Module):
    def __init__(self, input_size, image_size):
        super().__init__()
        print(input_size, image_size)
        # self.class_begin = nn.Sequential(
        #     nn.Linear(input_size, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 32),
        # )
        # self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=8, batch_first=True)
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
            nn.Linear(128 * 3, 128*3),
            nn.LeakyReLU(),
            nn.Linear(128*3, 128),
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


def train(args):
    shape_data, color_data, shape_classification, color_classification = load_data(args)
    save_data(args, (shape_data, color_data, shape_classification, color_classification))
    print("loaded image data...")

    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(shape_data, shape_classification, test_size=0.1, random_state=42)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(color_data, color_classification, test_size=0.1, random_state=42)
    print("Loaded train/test split data at rate of 0.1")

    train_dataset = TensorDataset(X_train0, X_train1, Y_train0, Y_train1)
    test_dataset = TensorDataset(X_test0, X_test1, Y_test0, Y_test1)

    train_loader = DataLoader(train_dataset, batch_size=int(len(X_train0) / 8), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=int(len(X_train0) / 8), shuffle=False)
    print("Finished loading dataset")

    print(shape_data.shape)
    print(color_data.shape)

    model = ContourClassifier(shape_data.shape[1], color_data.shape[1])
    model.train()
    if Path(args.model).exists():
        model = torch.load(args.model, weights_only=False)
        print(f"Loading model from {args.model}...")

    criterion_class_shape = nn.CrossEntropyLoss()
    criterion_class_color = nn.CrossEntropyLoss()
    optimizer_shape = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer_shape = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer_color = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    print("Image loading complete, beginning training...")

    should_exit = False
    for epoch in range(args.e):
        average_loss_shape = 0
        average_loss_color = 0
        try:
            runs = 0
            for batch_X_shape, batch_X_color, batch_Y_shape, batch_Y_color in train_loader:
                if args.s:
                    optimizer_shape.zero_grad()

                if args.c:
                    optimizer_color.zero_grad()

                class_pred, color_pred = model([batch_X_shape, batch_X_color])

                if args.s:
                    class_loss = criterion_class_shape(class_pred, batch_Y_shape)
                    average_loss_shape += class_loss.item()
                if args.c:
                    color_loss = criterion_class_color(color_pred, batch_Y_color)
                    average_loss_color += color_loss.item()
                runs += 1

                if args.s:
                    class_loss.backward()
                    optimizer_shape.step()
                if args.c:
                    color_loss.backward()
                    optimizer_color.step()

            print(f"Epoch {epoch + 1}, Loss Shape: {average_loss_shape / runs:.7f}, Loss Color: {average_loss_color / runs:.7f} {runs}")
        except KeyboardInterrupt:
            should_exit = True
            break

    torch.save(model, args.model)
    model.eval()
    print(f"Saved model as {args.model}")

    if should_exit:
        exit(0)

    correct_both = 0
    correct_shape = 0
    correct_color = 0
    incorrect = 0
    total = 0

    with torch.no_grad():
        for batch_X_shape, batch_X_color, batch_Y_shape, batch_Y_color in test_loader:
            class_pred, color_pred = model([batch_X_shape, batch_X_color])

            predicted_classes = torch.argmax(class_pred, dim=1)
            predicted_color = torch.argmax(color_pred, dim=1)

            for pred_shape, pred_color, shape_label, color_label in zip(predicted_classes, predicted_color, batch_Y_shape, batch_Y_color):
                if pred_shape == shape_label:
                    if pred_color == color_label:
                        correct_both += 1
                    else:
                        correct_shape += 1
                else:
                    if pred_color == color_label:
                        correct_color += 1
                    else:
                        incorrect += 1
            total += batch_Y_shape.size(0)

    accuracy = 100 * correct_both / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct Shape: {correct_shape}%")
    print(f"Correct Color: {correct_color}%")
    print(f"Incorrect: {incorrect}%")
    print(f"Total: {total}%")


def use(args):
    shape_data, color_data, shape_classification, color_classification = load_data(args)

    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(shape_data, shape_classification, test_size=0.1, random_state=42)
    X_train1, X_test1, Y_test1, Y_test1 = train_test_split(color_data, color_classification, test_size=0.1, random_state=42)

    test_dataset = TensorDataset(X_test0, X_test1, Y_test0, Y_test1)

    test_loader = DataLoader(test_dataset, batch_size=int(len(X_train1) / 16), shuffle=False)

    model = torch.load(args.model, weights_only=False)
    model.eval()

    correct = 0
    correct_color = 0
    correct_shape = 0
    incorrect = 0
    total = 0

    with torch.no_grad():
        for batch_X_shape, batch_X_color, batch_Y_shape, batch_Y_color in test_loader:
            class_pred, color_pred = model([batch_X_shape, batch_X_color])

            predicted_classes = torch.argmax(class_pred, dim=1)
            predicted_color = torch.argmax(color_pred, dim=1)

            print(class_pred)
            print(batch_Y_shape)
            print(predicted_classes)

            for pred_shape, pred_color, shape_label, color_label in zip(predicted_classes, predicted_color, batch_Y_shape, batch_Y_color):
                if pred_shape == shape_label:
                    if pred_color == color_label:
                        correct += 1
                    else:
                        correct_shape += 1
                else:
                    if pred_color == color_label:
                        correct_color += 1
                    else:
                        incorrect += 1
            total += batch_Y_shape.size(0)

    accuracy = 100 * correct / total
    print(f"Correct {correct}")
    print(f"Correct Color {correct_color}")
    print(f"Correct Shape {correct_shape}")
    print(f"Incorrect {incorrect}")
    print(f"Total {total}")
    print(f"Test Accuracy: {accuracy:.2f}%")

def use_camera(data_buffer, model, image):
    if image is None:
        return

    thresh = process.threshold_image(image)
    grey = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    bounds = process.find_bounds_and_contours(grey)

    if len(bounds) == 0:
        for buffer1, buffer2 in data_buffer:
            buffer1.clear()
            buffer2.clear()
        cv2.imshow("Window Me", image)
        return

    for i, data in enumerate(bounds):
        bound, con = data
        if i > len(data_buffer):
            continue
        buffer1, buffer2 = data_buffer[i]

        contour_features, con_len = preprocess_contour(con)
        x, y, w, h = bound
        color_data = preprocess_bounds(image, x, y, w, h)

        buffer1.append(contour_features)
        buffer2.append(color_data)


        shape_tensor = torch.tensor(np.array(list(buffer1)), dtype=torch.float32)
        color_tensor = torch.tensor(np.array(list(buffer2)), dtype=torch.float32)

        class_pred, color_pred = model([shape_tensor, color_tensor])

        predicted_classes = torch.argmax(class_pred, dim=1)
        predicted_color = torch.argmax(color_pred, dim=1)

        shape_prediction, count = torch.mode(predicted_classes)
        color_prediction, count = torch.mode(predicted_color)

        pred_class = "Unknown Shape"
        if shape_prediction == 0:
            pred_class = "T Shape"
        elif shape_prediction == 1:
            pred_class = "L Shape"
        elif shape_prediction == 2:
            pred_class = "Z Shape"

        pred_color = "Unknown Color"
        if color_prediction == 0:
            pred_color = "Gray Color"
        elif color_prediction == 1:
            pred_color = "Green Color"
        elif color_prediction == 2:
            pred_color = "Orange color"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(image, [con], -1, (255, 255, 255), 2)
        cv2.putText(image, "Predicted Class " + str(pred_class), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.putText(image, "Predicted Color " + str(pred_color), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imshow("Window Me", image)

def main():
    torch.set_num_threads(torch.get_num_threads())
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_parser = subparsers.add_parser("train", help="Train the network")

    train_parser.add_argument('-e', default=100, help="Epochs", type=int)
    train_parser.add_argument('-t', default="image_data.tmp.dat", help="Image data cache file", type=str)
    train_parser.add_argument('-s', action='store_true', default=False, help="Train shape classifier")
    train_parser.add_argument('-c', action='store_true', default=False, help="Train color classifier")
    train_parser.add_argument("database", help="Image database path")
    train_parser.add_argument("model", help="Model path")

    use_parser = subparsers.add_parser("use", help="Use the network")
    use_parser.add_argument('-t', default="image_data.tmp.dat", help="Image data cache file", type=str)
    use_parser.add_argument('-c', action='store_true', default=False, help="Use the network on the camera")
    use_parser.add_argument("database", help="Image database path")
    use_parser.add_argument("model", help="Model path")

    args = parser.parse_args()

    if args.mode == "train":
        print(torch.__config__.show())
        train(args)
    elif args.mode == "use":
        if args.c:
            model = torch.load(args.model, weights_only=False)
            model.eval()
            buffer_size = 120
            data_buffer = {}
            for i in range(0, 10):
                buffer1 = deque(maxlen=buffer_size)
                buffer2 = deque(maxlen=buffer_size)
                data_buffer[i] = (buffer1, buffer2)
            camera.open_camera("local/", lambda image, w, h: use_camera(data_buffer, model, image))
        else:
            use(args)


if __name__ == '__main__':
    main()
