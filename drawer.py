from typing import List, Tuple
import serial
import time
from svgpathtools import svg2paths, Path
import numpy as np
import math
import argparse

def extract_coordinates_from_svg(svg_file: str, segment_resolution: float = 1000) -> List[List[Tuple[float, float]]]:  
    paths, _ = svg2paths(svg_file) 
    all_coordinates: List[List[Tuple[float, float]]] = []

    xmin: float
    xmax: float 
    ymin: float 
    ymax: float
    for i, path in enumerate(paths):
        if i == 0:
            xmin, xmax, ymin, ymax = path.bbox()
        else:
            p_xmin, p_xmax, p_ymin, p_ymax = path.bbox()
            xmin = min(xmin, p_xmin)
            xmax = max(xmax, p_xmax)
            ymin = min(ymin, p_ymin)
            ymax = max(ymax, p_ymax)

    scale = max(xmax-xmin, ymax-ymin)

    cp: Path
    for path in paths:
        for cp in path.continuous_subpaths():
            num_points = segment_resolution*cp.length()/scale

            if cp.length()==math.inf or num_points<2:
                continue
            num_points = math.ceil(num_points)
            path_coordinates = []
            for i in range(num_points):
                t = i / (num_points - 1)  # Parameter t ranges from 0 to 1
                point = cp.point(t)
                x: float = point.real
                y: float = point.imag
                x -= xmin
                y -= ymin
                x /= scale
                y /= scale
                path_coordinates.append((x, y))
            all_coordinates.append(path_coordinates)

    return all_coordinates

def draw_points(ser: serial.Serial, coordinates: List[List[Tuple[float, float]]], feed: float = 1000):
    for path_coordinates in coordinates:
        first = True
        for x, y in path_coordinates:
            x *= 90 + 50
            y *= 90 + 50
            x += 5
            y += 5
            if first:
                pen_up(ser)
                move_to(ser, x, y)
                pen_down(ser)
            else:
                move_to_speed(ser, x, y, feed)
            first = False

    pen_up(ser)
    pen_up(ser)
    home(ser)
    pen_up(ser)
    pen_up(ser)
    pen_up(ser)
    pen_up(ser)

def svg(ser: serial.Serial, svg_file: str, segment_resolution: float, feed: float):
    def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        x1, y1 = p1
        x2, y2 = p2
        return (x1-x2)**2+(y1-y2)**2
    
    coordinates = extract_coordinates_from_svg(svg_file, segment_resolution)
    cur = 0
    path = [cur]
    totalDist = 0
    for i in range(1,len(coordinates)):
        dists = [(dist(coordinates[i][0],p[0]), pi) for (pi,p) in enumerate(coordinates) if pi != i and pi not in path]
        nextDist, cur = min(dists)
        totalDist += nextDist
        path.append(cur)
    draw_points(ser, coordinates, feed)

def g_command(ser: serial.Serial, cmd: str):
    print(cmd)
    if ser==None:
        return
    ser.write(cmd)
    while True:
        line = ser.readline()
        if line == b'ok\n':
            return
        print(line)

def home(ser: serial.Serial):
    g_command(ser, b'G28 X\n')
    g_command(ser, b'G28 Y\n')
    g_command(ser, b'G28 Z\n')
    g_command(ser, b'G91\n')#relative positioning
    g_command(ser, b'G0 Z2.8\n')
    g_command(ser, b'G90\n')#absolute positioning

def pen_up(ser: serial.Serial):
    g_command(ser, b'G91\n')#relative positioning
    g_command(ser, b'G0 Z2\n')
    g_command(ser, b'G90\n')#absolute positioning

def pen_down(ser: serial.Serial):
    g_command(ser, b'G91\n')#relative positioning
    g_command(ser, b'G0 Z-2\n')
    g_command(ser, b'G90\n')#absolute positioning

def move_to(ser: serial.Serial, x, y):
    g_command(ser, f'G0 X{x:.3f} Y{y:.3f}\n'.encode())

def move_to_speed(ser: serial.Serial, x, y, speed):
    g_command(ser, f'G0 X{x:.3f} Y{y:.3f} F{speed}\n'.encode())

def printer_init(ser: serial.Serial):
    while(ser.readable()):
        line = ser.readline()
        if line==b'ok\n' or line ==b'':
            break
        print(line)

    pen_up(ser)
    pen_up(ser)
    pen_up(ser)
    home(ser)
    pen_down(ser)

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("path", type=str, help="Required file path")
    parser.add_argument("--feed_speed", type=float, default=1000.0, help="Optional feed speed (default: 1000.0)")
    parser.add_argument("--serial", type=str, help="Optional serial port path")
    parser.add_argument("--baud", type=int, default=115200, help="Optional baud rate (default: 115200)")
    parser.add_argument("--segment_resolution", type=float, default=1000, help="Optional line segment resolution (default: 1000)")
    args = parser.parse_args()


    try:
        ser = None
        if args.serial:
            ser = serial.Serial(args.serial, args.baud, timeout=2)
            time.sleep(2)
            printer_init(ser)
        svg(ser, args.path, args.feed_speed, args.segment_resolution)
    except serial.SerialException as e:
        print(f"{e}")
    finally:
        if 'ser' in locals() and ser != None and ser.is_open:
            ser.close()
            print("Closed")


if __name__ == "__main__":
    main()