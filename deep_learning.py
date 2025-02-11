import faulthandler

import cv2
import argparse
import numpy as np
from pathlib import Path
import pickle
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import feed_forward as ff

NETWORK_INPUT_SIZE = 64

class CNNShapeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(2704, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, ff.CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Network:
    def __init__(self, model_file, database_path, batch_size = 4096):
        self.model_file = model_file
        if Path(model_file).exists():
            self.net = torch.load(model_file)
        else:
            self.net = CNNShapeNetwork()
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=200, factor=0.1)
        self.connection = sqlite3.connect(database_path)
        self.cursor = self.connection.cursor()
        self.batch_size = batch_size
        self.batch_images = np.empty((batch_size, NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), dtype=np.float32)
        self.batch_class = np.empty((batch_size,), dtype=np.longlong)


    def load_images(self):
        print("Loading new image batch")
        self.cursor.execute(
            "SELECT i.filename, i2.filename, b.x, b.y, b.width, b.height FROM extra_images i INNER JOIN images i2 ON i2.id = i.image_id" +
            " INNER JOIN bounds b ON i.image_id = b.image_id ORDER BY RANDOM() LIMIT ?", (self.batch_size,))
        data = self.cursor.fetchall()

        for i, image_data in enumerate(data):
            extra_filename, base_filename, x, y, w, h = image_data

            filename = base_filename
            if np.random.randint(0, 2) == 0:
                if Path(extra_filename).exists():
                    filename = extra_filename

            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image = ff.extract_image_from_bounds(image, x, y, w, h, max_size=4096)
            image = cv2.resize(image, (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)

            self.batch_images[i] = image
            self.batch_class[i] = ff.get_shape_class_from_filename(filename)
        print("Loading of images complete")

    def train(self, epochs, batch_process = 16):
        self.net.train()
        for epoch in range(epochs):
            self.load_images()
            print(f"Begin epoch {epoch}")
            try:
                runs = 0
                average_loss = 0
                index = 0

                while index < self.batch_size:
                    self.optimizer.zero_grad()
                    end_index = min(self.batch_size, index + batch_process)
                    image_data = torch.tensor(self.batch_images[index:end_index], dtype=torch.float32)
                    image_data = image_data.unsqueeze(1)
                    class_data = torch.tensor(self.batch_class[index:end_index], dtype=torch.long)

                    pred = self.net(image_data)

                    loss = self.criterion(pred, class_data)
                    average_loss += loss.item()
                    runs += 1

                    loss.backward()
                    self.optimizer.step()

                    index = end_index
                    print(f"Ran mini-batch with loss: {loss.item()}")

                self.scheduler.step(average_loss)
                print(f"Average loss for epoch {epoch} was {average_loss / runs}")
            except KeyboardInterrupt:
                break

        torch.save(self.net, self.model_file)
        print(f"Saved model as {self.model_file}")

    def __del__(self):
        self.connection.commit()
        self.connection.close()


def main():
    faulthandler.enable()
    print(f"Using {torch.get_num_threads()} threads")
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="mode", required=True)

    trainer = subparser.add_parser("train", help="Train the network")
    trainer.add_argument("database", help="Database with image data in it")
    trainer.add_argument("model", help="Model file to store the network")
    trainer.add_argument("--epochs", "-e", type=int, default=10000, help="Number of epochs to train")

    args = parser.parse_args()

    network = Network(args.model, args.database)

    if args.mode == "train":
        network.train(args.epochs)


if __name__ == "__main__":
    main()
