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
import camera
import process
import perf

NETWORK_INPUT_SIZE = 128

profiler = perf.PerfCounter()

class CNNShapeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5))
        self.pool = nn.MaxPool2d((2, 2), (2, 2))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(26912, 128)
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
    def __init__(self, model_file, database_path, batch_size = 4096, image_type = False):
        self.model_file = model_file
        if Path(model_file).exists():
            self.net = torch.load(model_file, weights_only=False)
        else:
            self.net = CNNShapeNetwork()
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=20, factor=0.1)
        if database_path is not None:
            self.connection = sqlite3.connect(database_path)
            self.cursor = self.connection.cursor()
        else:
            self.connection = None
            self.cursor = None
        self.batch_size = batch_size
        self.batch_images = np.empty((batch_size, NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), dtype=np.float32)
        self.batch_class = np.empty((batch_size,), dtype=np.longlong)
        self.image_type = image_type


    def load_images(self):
        print("Loading new image batch")
        event = profiler.start("Database Execute")
        if self.image_type:
            self.cursor.execute(
                "SELECT i.filename, b.x, b.y, b.width, b.height FROM images i" +
                " INNER JOIN bounds b ON i.id = b.image_id ORDER BY RANDOM() LIMIT ?", (self.batch_size,))
        else:
            self.cursor.execute(
                "SELECT i.filename, i2.filename, b.x, b.y, b.width, b.height FROM extra_images i INNER JOIN images i2 ON i2.id = i.image_id" +
                " INNER JOIN bounds b ON i.image_id = b.image_id ORDER BY RANDOM() LIMIT ?", (self.batch_size,))
        event.end()
        event = profiler.start("Database Fetchall")
        data = self.cursor.fetchall()
        event.end()

        event = profiler.start("Load Images from Database")
        for i, image_data in enumerate(data):
            if self.image_type:
                base_filename, x, y, w, h = image_data
            else:
                extra_filename, base_filename, x, y, w, h = image_data

            filename = base_filename
            if not self.image_type:
                if np.random.randint(0, 2) == 0 and Path(extra_filename).exists():
                    filename = extra_filename

            event2 = profiler.start("Image Load")
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            event2.end()
            event2 = profiler.start("Bounds Extract")
            image = ff.extract_image_from_bounds(image, x, y, w, h, max_size=4096)
            event2.end()
            event2 = profiler.start("Image Resize")
            image = cv2.resize(image, (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)

            self.batch_images[i] = image
            self.batch_class[i] = ff.get_shape_class_from_filename(filename)
            event2.end()
        event.end()
        print("Loading of images complete")

    def train(self, epochs, batch_process = 16):
        self.net.train()
        for epoch in range(epochs):
            event = profiler.start("Epoch")
            try:
                self.load_images()
                print(f"Begin epoch {epoch}")
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

                self.scheduler.step(average_loss)
                print(f"Average loss for epoch {epoch} was {average_loss / runs}")
            except KeyboardInterrupt:
                break
            event.end()

        torch.save(self.net, self.model_file)
        print(f"Saved model as {self.model_file}")

    def test(self, test_size = 4096):
        self.net.eval()
        correct = 0
        incorrect = 0
        while test_size > 0:
            self.load_images()
            print(f"Begin testing phase, images remaining: {test_size}")
            images = torch.tensor(self.batch_images[:min(test_size, self.batch_size)], dtype=torch.float32)
            images = images.unsqueeze(1)
            classes = torch.tensor(self.batch_class[:min(test_size, self.batch_size)], dtype=torch.long)

            pred = self.net(images)

            pred_class = torch.argmax(pred, dim=1)

            for pred, actual in zip(pred_class, classes):
                if pred == actual:
                    correct += 1
                else:
                    incorrect += 1

            test_size -= self.batch_size

        print(f"Correct: {correct} / Incorrect: {incorrect}")
        print(f"Test set accuracy: {correct / (correct + incorrect)}")

    def run_model_on_image(self, image):
        if image is None:
            return
        thresh = process.threshold_image(image.copy())
        bounds = process.find_bounds_and_contours(cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY))
        if len(bounds) == 0:
            cv2.imshow("Hello", image)
            return

        draw = image.copy()
        print ("Running model on camera input")
        images = []
        for bound, con in bounds:
            x,y,w,h = bound
            bounded_image = ff.extract_image_from_bounds(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), x, y, w, h, max_size=4096)
            bounded_image = cv2.resize(bounded_image, (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
            bounded_image = torch.tensor(bounded_image, dtype=torch.float32)
            bounded_image = bounded_image.unsqueeze(0)
            images.append(bounded_image)

        images = torch.tensor(np.array(images), dtype=torch.float32)
        pred = self.net(images)

        pred_class = torch.argmax(pred, dim=1)

        for pred_class, bound_and_con in zip(pred_class, bounds):
            bound, con = bound_and_con
            x,y,w,h = bound

            shape = "Unknown"
            if pred_class == 0:
                shape = "T Shape"
            elif pred_class == 1:
                shape = "L Shape"
            elif pred_class == 2:
                shape = "Z Shape"
            print(f"\tFound shape {shape}")

            cv2.rectangle(draw, (x, y), (x + w, y + h), color=(0, 255, 0))
            cv2.drawContours(draw, [con], -1, (255, 255, 255), 2)
            cv2.putText(draw, shape, (x, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow("Hello", cv2.hconcat(draw, thresh))

    def camera(self, c):
        self.net.eval()
        print(f"Opening camera {c}")
        camera.open_camera("local", lambda image, w, h: self.run_model_on_image(image), c)

    def __del__(self):
        if self.connection is not None:
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
    trainer.add_argument("--batch-size", '-b', dest='b', type=int, default=4096, help="Batch size")
    trainer.add_argument("-d", action='store_true', default=False, help="Use the normal database instead of the deep learning one")

    tester = subparser.add_parser("test", help="Test the network")
    tester.add_argument("database", help="Database with image data in it")
    tester.add_argument("model", help="Model file to store the network")
    tester.add_argument("--test-size", '-t', dest='t', type=int, default=4096, help="Number of images to test")
    tester.add_argument("--batch-size", '-b', dest='b', type=int, default=4096, help="Batch size")
    tester.add_argument("-d", action='store_true', default=False, help="Use the normal database instead of the deep learning one")

    camera_parser = subparser.add_parser("camera", help="Test using input from a camera")
    camera_parser.add_argument("model", help="Model file to store the network")
    camera_parser.add_argument("--database", dest="database", default=None, required=False, help="Unused")
    camera_parser.add_argument("--batch-size", '-b', dest='b', type=int, default=4096, help="Unused")
    camera_parser.add_argument("-d", action='store_true', default=False, help="Unused")
    camera_parser.add_argument('-c', action='store', default=0, help="Camera to use")

    args = parser.parse_args()

    network = Network(args.model, args.database, args.b, args.d)

    if args.mode == "train":
        network.train(args.epochs)
    elif args.mode == "test":
        network.test(args.t)
    elif args.mode == "camera":
        network.camera(args.c)

    profiler.print_profiles()


if __name__ == "__main__":
    main()
