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
    "base": 1600,
    "shoulder": 1550,
    "elbow": 1350,
    "wrist": 1300,
    "rotate": 1400,
    "grip": 1000,
}

current_positions = home_positions.copy()

current_cube = 0

open_grip = 1000
close_grip = 1400

static_positions = {
    "square": {
        "base": 1600,
        "shoulder": 1550,
        "elbow": 1350,
        "wrist": 1300,
        "rotate": 1400,
        "grip": 1000,
    },
    "depot": {
        "base": 1150,
        "shoulder": 1550,
        "elbow": 1350,
        "wrist": 1300,
        "rotate": 1400,
        "grip": 1000,
    },
    "c0": {
        "base": 1190,
        "shoulder": 1380,
        "elbow": 1460,
        "wrist": 650,
        "rotate": 1400,
        "grip": 1000,
    },
    "c1": {
        "base": 1100,
        "shoulder": 1280,
        "elbow": 1320,
        "wrist": 620,
        "rotate": 1400,
        "grip": 1000,
    },
    "c2": {
        "base": 1040,
        "shoulder": 1160,
        "elbow": 1090,
        "wrist": 650,
        "rotate": 1400,
        "grip": 1000,
    },
    "c3": {
        "base": 1200,
        "shoulder": 1070,
        "elbow": 950,
        "wrist": 660,
        "rotate": 1400,
        "grip": 1000,
    },
    "x1": home_positions,
    "x2": home_positions,
    "x3": home_positions,
    "x4": home_positions,
    "o1": home_positions,
    "o2": home_positions,
    "o3": home_positions,
    "o4": home_positions,
    "s0_0": {
        "base": 1850,
        "shoulder": 1400,
        "elbow": 1480,
        "wrist": 690,
        "rotate": 1400,
        "grip": 1400,
    },
    "s0_1": {
        "base": 1650,
        "shoulder": 1460,
        "elbow": 1570,
        "wrist": 660,
        "rotate": 1400,
        "grip": 1000,
    },
    "s0_2": {
        "base": 1380,
        "shoulder": 1450,  # may need to increase by 10-30
        "elbow": 1570,
        "wrist": 660,
        "rotate": 1400,
        "grip": 1000,
    },
    "s1_0": {
        "base": 1750,
        "shoulder": 1190,
        "elbow": 1180,
        "wrist": 690,
        "rotate": 1400,
        "grip": 1400,
    },
    "s1_1": {
        "base": 1620,
        "shoulder": 1220,
        "elbow": 1240,
        "wrist": 660,
        "rotate": 1400,
        "grip": 1000,
    },
    "s1_2": {
        "base": 1400,
        "shoulder": 1220,
        "elbow": 1240,
        "wrist": 660,
        "rotate": 1400,
        "grip": 1400,
    },
    "s2_0": {
        "base": 1720,
        "shoulder": 1035,
        "elbow": 920,
        "wrist": 760,
        "rotate": 1400,
        "grip": 1400,
    },
    "s2_1": {
        "base": 1580,
        "shoulder": 1040,
        "elbow": 940,
        "wrist": 760,
        "rotate": 1400,
        "grip": 1000,
    },
    "s2_2": {
        "base": 1460,
        "shoulder": 1100,
        "elbow": 1060,
        "wrist": 760,
        "rotate": 1400,
        "grip": 1000,
    },
    "accept": {
        "base": 1500,
        "shoulder": 1350,
        "elbow": 1100,
        "wrist": 1300,
        "rotate": 1400,
        "grip": 1000,
    },
}

board_state = [["" for _ in range(3)] for _ in range(3)]


class SerialController:
    def __init__(self, serial_port, baud):
        try:
            self.serial = serial.Serial(serial_port, baud)
            print(f"Connected to {serial_port}!")
        except serial.SerialException as e:
            print(f"Failed to open {serial_port}: {e}")
            self.serial = None

    def read(self, amount):
        if self.serial is None:
            return ""
        return self.serial.read(amount)

    def wait_for_movement(self):
        if self.serial is None:
            return
        while True:
            self.write("Q \r")
            byte = self.read(1)
            if byte == b'.':
                break

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
    def __init__(self, serial_port, baud, increment_amount=10):
        self.current_joint_index = 0
        self.row = -1
        self.col = -1
        self.current_cube = 0
        self.game_running = False
        self.game_setup = False
        self.player_turn = False
        self.taken_spaces = []
        self.player_spaces = []
        self.increment_amount = increment_amount
        self.serial = SerialController(serial_port, baud)
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
        self.tic_tac_toe_canvas.bind("<Button-1>", self.on_tic_tac_toe_click)

        # Draw grid lines
        self.offset = 25
        self.cell_size = (tick_width - self.offset) / 3
        cell_size = self.cell_size
        for i in range(1, 3):
            self.tic_tac_toe_canvas.create_line(i * cell_size, self.offset, i * cell_size, tick_height - self.offset,
                                                width=2)
            self.tic_tac_toe_canvas.create_line(self.offset, i * cell_size, tick_width - self.offset, i * cell_size,
                                                width=2)

        self.control_frame = tk.Frame(self.root, padx=200, pady=150)
        self.control_frame.grid(row=1, column=1, sticky="se")

        self.selected_joint_label = tk.Label(self.control_frame, text="Selected Joint: Base", font=("Arial", 12))
        self.selected_joint_label.pack()

        self.joint_value_label = tk.Label(self.control_frame, text=current_positions[self.selected_joint],
                                          font=("Arial", 14, "bold"))
        self.joint_value_label.pack()

        self.btn_decrease = tk.Button(self.control_frame, text="−", font=("Arial", 14), command=self.decrease_value)
        self.btn_decrease.pack(side="left", padx=5)

        self.btn_increase = tk.Button(self.control_frame, text="+", font=("Arial", 14), command=self.increase_value)
        self.btn_increase.pack(side="right", padx=5)

        self.btn_update = tk.Button(self.control_frame, text="Update", font=("Arial", 14), command=self.send_positions)
        self.btn_update.pack(pady=10)

        self.input_frame = tk.Frame(self.root)
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.label = tk.Label(self.input_frame, text="Enter command:")
        self.label.grid(row=0, column=0, padx=5, pady=5)

        self.entry = tk.Entry(self.input_frame)
        self.entry.grid(row=1, column=0, padx=5, pady=5)

        self.run_button = tk.Button(self.input_frame, text="Execute", command=self.run_function)
        self.run_button.grid(row=2, column=0, padx=5, pady=5)

        self.open_grip_button = tk.Button(self.input_frame, text="Open Grip", command=self.open_grip)
        self.open_grip_button.grid(row=3, column=0, padx=5, pady=5)

        self.close_grip_button = tk.Button(self.input_frame, text="Close Grip", command=self.close_grip)
        self.close_grip_button.grid(row=3, column=1, padx=5, pady=5)

        self.accept_piece_button = tk.Button(self.input_frame, text="Accept Piece", command=self.accept_piece)
        self.accept_piece_button.grid(row=4, column=0, padx=5, pady=5)

        self.start_game_button = tk.Button(self.input_frame, text="Start Game", command=self.setup_start_game)
        self.start_game_button.grid(row=5, column=0, padx=5, pady=5)

        self.take_cube_button = tk.Button(self.input_frame, text="Take Cube", command=self.take_cube)
        self.take_cube_button.grid(row=5, column=1, padx=5, pady=5)

        self.cleanup_cubes_button = tk.Button(self.input_frame, text="Cleanup", command=self.cleanup)
        self.cleanup_cubes_button.grid(row=5, column=2, padx=5, pady=5)

        self.message_label = tk.Label(self.input_frame, text="")
        self.message_label.grid(row=6, column=0, padx=5, pady=5)

        self.root.bind("<w>", self.increase_value)
        self.root.bind("<s>", self.decrease_value)
        self.root.bind("<a>", self.previous_joint)
        self.root.bind("<d>", self.next_joint)
        self.root.bind("<p>", self.print_positions)
        self.root.bind("<h>", self.return_home)
        self.root.bind("<c>", self.close_grip)
        self.root.bind("<o>", self.open_grip)
        self.root.bind("<j>", self.square)
        self.root.bind("<k>", self.depot)
        self.root.bind("<t>", self.take_cube)
        self.root.bind("<Return>", self.button_return)

        self.root.bind("<Button-1>", self.on_click)

        self.root.mainloop()

    def setup_start_game(self, event=None):
        if self.game_setup:
            cleanup()
        self.game_setup = False
        self.game_running = True
        self.current_cube = 0
        self.player_turn = random.choice([True, False])
        self.taken_spaces = []
        self.player_spaces = []
        self.play_game()

    def play_game(self, event=None):
        if self.current_cube <= 3 and not self.game_setup:
            self.message_label.config(text="Please provide the next cube!")
            self.accept_piece()
            return
        if self.current_cube < 0:
            self.message_label.config(text="Game Over! No more cubes left!")
            self.current_cube = 0
            self.game_running = False
            return
        if self.player_turn:
            self.message_label.config(text="It's your turn!")
            return
        self.message_label.config(text="It's robot's turn!")
        while True:
            p1 = random.randint(0, 2)
            p2 = random.randint(0, 2)
            position = f"s{p1}_{p2}"
            if not position in self.taken_spaces:
                break
        self.piece_to_square(self.current_cube, p1, p2)
        self.taken_spaces.append(position)
        self.current_cube -= 1
        self.draw_o(p1, p2)
        self.player_turn = True
        self.play_game()

    def cleanup(self, event=None):
        for square in self.taken_spaces:
            if square in self.player_spaces:
                continue
            self.square_to_piece_ul(self.current_cube, square)
            self.current_cube += 1

    def take_cube(self, event=None):
        if self.current_cube > 3:
            self.message_label.config(text="No more cubes positions left!")
            return
        self.close_grip()
        self.message_label.config(text=f"Moving cube to position {self.current_cube}")
        self.set_to(static_positions[f"c{self.current_cube}"])
        self.current_cube += 1
        if self.current_cube == 4:
            self.game_setup = True
            self.current_cube = 3
            self.message_label.config(text="Cubes are in position!")
        self.open_grip()
        if self.game_running:
            self.play_game()

    def on_click(self, event):
        widget = event.widget
        if widget != self.entry:
            self.root.focus_set()

    def button_return(self, event=None):
        if self.entry.focus_get() == self.entry:
            self.root.focus_set()
            self.run_function()
            return
        self.send_positions()

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

    def on_tic_tac_toe_click(self, event):
        self.row, self.col = (self.offset + event.y) // self.cell_size, (self.offset + event.x) // self.cell_size
        if self.player_turn:
            #self.piece_to_square(self.current_cube, int(self.row), int(self.col))
            self.draw_x(self.row, self.col)
            self.taken_spaces.append(f"s{int(self.row)}_{int(self.col)}")
            self.player_spaces.append(f"s{int(self.row)}_{int(self.col)}")
            #self.current_cube -= 1
            self.player_turn = False
            self.play_game()
        print(f"{self.row}, {self.col}")

    def draw_x(self, row, col):
        # Calculate the pixel coordinates for the top-left and bottom-right of the cell
        x1 = self.offset + col * self.cell_size
        y1 = self.offset + row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size

        # Draw two diagonal lines for an X
        self.tic_tac_toe_canvas.create_line(x1, y1, x2, y2, width=3, fill="red")
        self.tic_tac_toe_canvas.create_line(x1, y2, x2, y1, width=3, fill="red")

    def draw_o(self, row, col):
        # Calculate the pixel coordinates for the cell
        x1 = self.offset + col * self.cell_size
        y1 = self.offset + row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size

        # Draw an oval for an O
        self.tic_tac_toe_canvas.create_oval(x1, y1, x2, y2, width=3, outline="blue")

    def increase_value(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        current_positions[self.selected_joint] = current_positions[self.selected_joint] + self.increment_amount
        self.update_values()

    def decrease_value(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        current_positions[self.selected_joint] = current_positions[self.selected_joint] - self.increment_amount
        self.update_values()

    def next_joint(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        self.current_joint_index = (self.current_joint_index - 1) % len(fields)
        self.selected_joint = list(current_positions.keys())[self.current_joint_index]
        self.update_values()

    def previous_joint(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        self.current_joint_index = (self.current_joint_index + 1) % len(fields)
        self.selected_joint = list(current_positions.keys())[self.current_joint_index]
        self.update_values()

    def update_values(self):
        if self.entry.focus_get() == self.entry:
            return
        label = self.selected_joint
        label = uppercase_char_in_string(label, 0)
        self.selected_joint_label.config(text=f"Selected Joint: {label}")
        self.joint_value_label.config(text=f"{current_positions[self.selected_joint]}")
        x, y = joint_positions[self.selected_joint]
        self.canvas.coords(self.selection, x - 15, y - 15, x + 15, y + 15)

    def print_positions(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        print("{")
        for key, value in current_positions.items():
            print(f"\t\"{key}\": {value},")
        print("}")
        print()

    def return_home(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        self.set_to(home_positions)

    def open_grip(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        print("Opening grip!")
        current_positions["grip"] = open_grip
        self.send_positions()

    def close_grip(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        print("Closing grip!")
        current_positions["grip"] = close_grip
        self.send_positions()

    def set_to(self, positions):
        if self.entry.focus_get() == self.entry:
            return
        global current_positions

        # grip = current_positions["grip"]
        # current_positions =
        # current_positions["grip"] = grip

        copy_positions = positions.copy()
        copy_positions["grip"] = current_positions["grip"]
        # if not (current_positions["elbow"] == static_positions["depot"]["elbow"]):
        current_positions["elbow"] = static_positions["depot"]["elbow"]
        # self.send_positions()
        # if not (current_positions["shoulder"] == static_positions["depot"]["shoulder"]):
        current_positions["shoulder"] = static_positions["depot"]["shoulder"]
        # self.send_positions()
        # if not (current_positions["wrist"] == static_positions["depot"]["wrist"]):
        current_positions["wrist"] = static_positions["depot"]["wrist"]
        self.send_positions()

        current_positions["base"] = 800
        self.send_positions()

        current_positions["base"] = copy_positions["base"]
        self.send_positions()

        if not (current_positions["wrist"] == copy_positions["wrist"]):
            current_positions["wrist"] = copy_positions["wrist"]
            self.send_positions()

        self.serial.write_positions(current_positions)
        self.serial.wait_for_movement()
        current_positions = copy_positions

        self.send_positions()

    def home(self, event=None):
        self.set_to(home_positions)

    def square(self, event=None):
        self.set_to(static_positions["square"])

    def depot(self, event=None):
        self.set_to(static_positions["depot"])

    def accept_piece(self, event=None):
        self.set_to(static_positions["accept"])
        self.open_grip()

    def piece_to_square(self, piece, square_x, square_y):
        if self.entry.focus_get() == self.entry:
            return
        # self.depot()
        self.set_to(static_positions[f"c{piece}"])
        self.close_grip()
        # self.depot()
        # self.square()
        self.set_to(static_positions[f"s{square_x}_{square_y}"])
        self.open_grip()
        # self.square()

    def square_to_piece_ul(self, piece, position):
        if self.entry.focus_get() == self.entry:
            return
        # self.square()
        self.set_to(static_positions[position])
        self.close_grip()
        # self.square()
        # self.depot()
        self.set_to(static_positions[f"c{piece}"])
        self.open_grip()
        # self.depot()

    def square_to_piece(self, piece, square_x, square_y):
        self.square_to_piece_ul(piece, f"s{square_x}_{square_y}")

    def send_positions(self, event=None):
        if self.entry.focus_get() == self.entry:
            return
        self.update_values()
        self.serial.write_positions(current_positions)
        self.serial.wait_for_movement()

    def run_function(self, event=None):
        text = self.entry.get()
        print(f"Running command {text}")
        parts = text.split(" ")
        print(parts)
        if parts[0] == "c":
            self.piece_to_square(parts[1], parts[2], parts[3])
        elif parts[0] == "s":
            self.square_to_piece(parts[1], parts[2], parts[3])
        elif parts[0] == "g":
            self.set_to(static_positions[parts[1]])
        elif parts[0] == "accept":
            self.accept_piece()


def main():
    interface = UserInterface("COM1", 115200)


if __name__ == "__main__":
    main()
