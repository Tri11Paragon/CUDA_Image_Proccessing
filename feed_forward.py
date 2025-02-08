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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

MAX_POINTS = 255
CLASSES = 3 * 3

def preprocess_contour(contour):
    contour = np.array(contour, dtype=np.float32)

    # Normalize to [0,1]
    contour -= contour.min(axis=0)
    contour /= (contour.max(axis=0) + 1e-6)

    # Flatten and pad
    contour = contour.flatten()
    padded = np.zeros(MAX_POINTS * 2 + 3, dtype=np.float32)
    length = min(len(contour), MAX_POINTS * 2)
    padded[:length] = contour[:length]

    return padded

def get_class_from_filename(path):
    parts = path.split('_')
    color = parts[1]
    t = parts[2]
    if t == "t":
        if color == "grey":
            return 0
        elif color == "green":
            return 1
        elif color == "orange":
            return 2
    elif t == 'l':
        if color == "grey":
            return 3
        elif color == "green":
            return 4
        elif color == "orange":
            return 5
    elif t == 't':
        if color == "grey":
            return 6
        elif color == "green":
            return 7
        elif color == "orange":
            return 8
    return 0

def load_data_from_db(args):
    conn = sqlite3.connect(args.database)
    cursor = conn.cursor()

    cursor.execute("SELECT c.contour, i.filename, b.x, b.y, b.width, b.height FROM contours c INNER JOIN images i ON c.image_id = i.id INNER JOIN bounds b ON i.id = b.image_id")
    data = cursor.fetchall()
    conn.close()

    X, Y = [], []

    for contour_blob, image_path, x, y, w, h in data:
        contour = pickle.loads(contour_blob)
        image = cv2.imread(image_path)

        contour_features = preprocess_contour(contour)
        b, g, r = process.extract_color(image, contour)

        contour_features[-1] = b / 255.0
        contour_features[-2] = g / 255.0
        contour_features[-3] = r / 255.0

        X.append(contour_features)
        Y.append(get_class_from_filename(image_path))

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.long)

class ContourClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.classifier = nn.Softmax(1)

    def forward(self, x):
        features = self.fc(x)
        class_output = self.classifier(features)
        return class_output

def train(args):
    X, Y = load_data_from_db(args)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X.shape[1]
    num_classes = CLASSES
    model = ContourClassifier(input_size, num_classes)

    criterion_class = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Image loading complete, beginning training")

    should_exit = False
    for epoch in range(args.e):
        try:
            average_loss = 0
            runs = 0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                class_pred = model(batch_X)

                loss = criterion_class(class_pred, batch_Y)
                average_loss += loss.item()
                runs += 1

                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {average_loss / runs:.4f}, {runs}")
        except KeyboardInterrupt:
            should_exit = True
            break

    torch.save(model.state_dict(), args.model)
    model.eval()
    print(f"Saved model as {args.model}")

    if should_exit:
        exit(0)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            class_pred = model(batch_X)

            print(class_pred)
            predicted_classes = torch.argmax(class_pred, dim=1)

            correct += (predicted_classes == batch_Y).sum().item()
            total += batch_Y.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def use(args):
    X, Y = load_data_from_db(args)
    dataset = TensorDataset(X, Y)

    input_size = X.shape[1]
    num_classes = CLASSES

    model = ContourClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(args.model))
    model.eval()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    train_parser = subparsers.add_parser("train", help="Train the network")

    train_parser.add_argument('-e', default=100, help="Epochs", type=int)
    train_parser.add_argument("database", help="Image database path")
    train_parser.add_argument("model", help="Model path")

    use_parser = subparsers.add_parser("use", help="Use the network")
    use_parser.add_argument("database", help="Image database path")
    use_parser.add_argument("model", help="Model path")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "use":
        use(args)

if __name__ == '__main__':
    main()