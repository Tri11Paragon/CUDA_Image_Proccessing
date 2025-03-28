import serial
import tkinter as tk
from PIL import Image, ImageTk
import random

def uppercase_char_in_string(s, index):
    if index < 0 or index >= len(s):
        raise ValueError("Index out of range")
    return s[:index] + s[index].upper() + s[index + 1:]

fields = {
    "base": "#0 P",
    "shoulder": "#1 P",
    "elbow": "#2 P",
    "wrist": "#3 P",
    "rotate": "#4 P",
    "grip": "#5 P"
}

joint_positions = {
    "base": (376, 383),
    "shoulder": (402, 289),
    "elbow": (400, 90),
    "wrist": (163, 126),
    "rotate": (110, 214),
    "grip": (66, 313)
}

home_positions = {
    "base": 1500,
    "shoulder": 2150,
    "elbow": 2150,
    "wrist": 1300,
    "rotate": 1400,
    "grip": 1000
}

current_positions = home_positions.copy()

class SerialController:
    def __init__(self, serial_port):
        try:
            self.serial = serial.Serial(serial_port, 115200, timeout=1)
        except serial.SerialException as e:
            print(f"Failed to open {serial_port}: {e}")
            self.serial = None

    def write(self, string):
        if self.serial is None:
            return
        self.serial.write((string + "\r").encode())

    def write_positions(self, positions):
        string = ""
        for key, value in positions.items():
            string += f"{fields[key]} {value}"
        string += " T1000"
        self.write(string)

    def home(self):
        self.write_positions(home_positions)


class UserInterface:
    def __init__(self, serial_port, increment_amount = 10):
        self.current_joint_index = 0
        self.increment_amount = increment_amount
        self.serial = SerialController(serial_port)
        self.root = tk.Tk()
        self.root.title("Robot Arm Controller")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=2)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)

        self.top_left_frame = tk.Frame(self.root)
        self.top_left_frame.grid(row=0, column=0, sticky="nsw")

        self.image = Image.open("arm.jpg")  # Replace with actual image path
        # self.image = self.image.resize((400, 300))
        img_tk = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(self.top_left_frame, width=640, height=480)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        self.highlight = self.canvas.create_oval(-50, -50, -50, -50, outline="yellow", width=3)
        self.selection = self.canvas.create_oval(-50, -50, -50, -50, fill="lime", width=3)
        self.highlighted_joint = None
        self.selected_joint = "base"
        x, y = joint_positions[self.selected_joint]
        self.canvas.coords(self.selection, x - 15, y - 15, x + 15, y + 15)

        self.canvas.bind("<Motion>", self.mouse_motion_event)
        self.canvas.bind("<Button-1>", self.button_event)

        self.top_right_frame = tk.Frame(self.root)
        self.top_right_frame.grid(row=0, column=1, sticky="ne")

        tick_width = 600
        tick_height = 600
        self.tic_tac_toe_canvas = tk.Canvas(self.top_right_frame, width=tick_width, height=tick_height, bg="white")
        self.tic_tac_toe_canvas.pack()

        # Draw grid lines
        offset = 25
        self.cell_size = (tick_width - offset) / 3
        cell_size = self.cell_size
        for i in range(1, 3):
            self.tic_tac_toe_canvas.create_line(i * cell_size, offset, i * cell_size, tick_height - offset, width=2)
            self.tic_tac_toe_canvas.create_line(offset, i * cell_size, tick_width - offset, i * cell_size, width=2)

        self.control_frame = tk.Frame(self.root, padx=200, pady=150)
        self.control_frame.grid(row=1, column=1, sticky="se")

        self.selected_joint_label = tk.Label(self.control_frame, text="Selected Joint: Base", font=("Arial", 12))
        self.selected_joint_label.pack()

        self.joint_value_label = tk.Label(self.control_frame, text=current_positions[self.selected_joint], font=("Arial", 14, "bold"))
        self.joint_value_label.pack()

        self.btn_decrease = tk.Button(self.control_frame, text="−", font=("Arial", 14), command=self.decrease_value)
        self.btn_decrease.pack(side="left", padx=5)

        self.btn_increase = tk.Button(self.control_frame, text="+", font=("Arial", 14), command=self.increase_value)
        self.btn_increase.pack(side="right", padx=5)

        self.btn_update = tk.Button(self.control_frame, text="Update", font=("Arial", 14), command=self.send_positions)
        self.btn_update.pack(pady=10)

        self.root.bind("<w>", self.increase_value)
        self.root.bind("<s>", self.decrease_value)
        self.root.bind("<a>", self.previous_joint)
        self.root.bind("<d>", self.next_joint)
        self.root.bind("<p>", self.print_positions)
        self.root.bind("<Return>", self.send_positions)

        self.root.mainloop()

    def mouse_motion_event(self, event):
        min_dist = 35  # Distance threshold for highlighting
        nearest_joint = None
        for joint, (x, y) in joint_positions.items():
            dist = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if dist < min_dist:
                nearest_joint = joint
                min_dist = dist

        if nearest_joint:
            x, y = joint_positions[nearest_joint]
            self.canvas.coords(self.highlight, x - 15, y - 15, x + 15, y + 15)
            self.highlighted_joint = nearest_joint
        else:
            self.canvas.coords(self.highlight, -50, -50, -50, -50)
            self.highlighted_joint = None

    def button_event(self, event):
        # print(f"ButtonEvent {event}")
        if self.highlighted_joint:
            self.selected_joint = self.highlighted_joint
            coords = self.canvas.coords(self.highlight)
            self.canvas.coords(self.selection, coords[0], coords[1], coords[2], coords[3])
            self.update_values()

    def increase_value(self, event=None):
        current_positions[self.selected_joint] = current_positions[self.selected_joint] + self.increment_amount
        self.update_values()

    def decrease_value(self, event=None):
        current_positions[self.selected_joint] = current_positions[self.selected_joint] - self.increment_amount
        self.update_values()

    def next_joint(self, event=None):
        self.current_joint_index = (self.current_joint_index - 1) % len(fields)
        self.selected_joint = list(current_positions.keys())[self.current_joint_index]
        self.update_values()

    def previous_joint(self, event=None):
        self.current_joint_index = (self.current_joint_index + 1) % len(fields)
        self.selected_joint = list(current_positions.keys())[self.current_joint_index]
        self.update_values()

    def update_values(self):
        label = self.selected_joint
        label = uppercase_char_in_string(label, 0)
        self.selected_joint_label.config(text=f"Selected Joint: {label}")
        self.joint_value_label.config(text=f"{current_positions[self.selected_joint]}")
        x, y = joint_positions[self.selected_joint]
        self.canvas.coords(self.selection, x - 15, y - 15, x + 15, y + 15)

    def print_positions(self, event=None):
        print("{")
        for key, value in current_positions.items():
            print(f"\t\"{key}\": {value}")
        print("}")
        print()

    def send_positions(self, event=None):
        self.serial.write_positions(current_positions)


def main():
    interface = UserInterface("COM1")

if __name__ == "__main__":
    main()