# YOLO Object Detection Controlled Car with Arduino

Author: Preston Ito  
Date: December 9, 2022

## Introduction and Motivation

For my final Electronics project, I coded and built a car that moves depending on what it sees with an external USB camera that is attached to it. The car uses YOLO object detection to scan what it sees, and if it sees a single person, it will move toward them. The car also has an LCD display that will show what it’s doing, whether it be turning left or right or moving forward. If it’s not moving (it won't move if it detects anything but a single person), the LCD will display what it sees and how many it sees. This project could be expanded to perform a wide range of robotic and autonomous functions, such as creating a robot vacuum to navigate a room, robot waiters in restaurants, or potentially even autonomous robots that can explore areas that could be hard to get to for humans. It could also even be used for less practical purposes, such as a cat or dog toy that could follow around a child.

## Background and Theory

This project can be broken down into two categories: hardware and software. The hardware includes the components used, and the software side includes the YOLO object detection code and the Arduino code. The basis of this project was inspired by a previous project, titled [Bluetooth Controlled Car](https://create.arduino.cc/projecthub/samanfern/bluetooth-controlled-car-d5d9ca).

### Hardware

The following parts were used in assembling this project:

- Arduino UNO
- Adafruit 16x2 RGB LCD Shield
- Gikfun Screw Shield Expansion Board for Arduino UNO
- 9V battery
- 9V battery to wires adapter
- ShangHJ 4 Sets DC Gearbox Motor Kit (includes 4 wheels, 4 motors, and motor driver)
- External USB camera
- Mini breadboard
- Various wires (jumper and normal)
- Wood frame
- Zip Ties
- Tape
- Computer with 2 USB ports

### Software

On the software side, two main pieces of code were used:

1. Arduino Code
2. YOLOv5 Object Detection Code

The Arduino code was primarily written by myself, with inspiration from the Bluetooth Controlled Car project. It involved receiving and interpreting information sent from a Python YOLOv5 program and performing different actions based on what was being read.

The YOLOv5 Object Detection Code is an open-source package created by Ultralytics. YOLO (You Only Look Once) is a fast method of object detection suitable for real-time applications like self-driving cars, robotics, and surveillance systems. YOLOv5 uses a neural network trained on the COCO dataset to detect objects and create bounding boxes around them. Although this code wasn't written by me, I still had to learn and be somewhat familiar with the functionality of the Python file in order to edit where necessary and send desired messages to the Arduinio file.

## Procedure

The project was developed in several phases:

### Phase 1: Testing Basic Functionality (Software)

- Ensuring YOLO worked with an external USB camera
- Executing the YOLO file in Visual Studio Code
- Identifying the specific blocks of code that outputs detected objects
- Establishing communication between the Python YOLO file and the Arduino

### Phase 2: Testing Basic Functionality (Hardware)

- Building the physical components and testing motors
- Attempting to connect the HC-05 Bluetooth module to control the car from a phone (found unsuccessful)
- Testing the LCD display to ensure it worked as intended

### Phase 3: Piecing Hardware and Software Together

- Sending messages from YOLO to the Arduino to control the LCD display
- Controlling the car's movement without Bluetooth
- Designing and laser cutting a frame for the car

### Phase 4: Fine Tuning and Improvements

- Making the car turn left or right depending on the detected person's position
- Fixing issues with the LCD display to show only one message at a time
- Making the display indicate what the car is doing (e.g., turning or moving forward)

## Results

The project allowed the car to move based on YOLO detections, and it successfully displayed messages on the LCD screen to indicate its actions. Although the code took long to load and didn't control the robot instantaneously, I was still happy with the product as it met the basic objectives.

## Analysis

While the project met the basic goals, some areas could be improved:

- Slowness in processing due to GPU limitations
- Timing issues between the Arduino and YOLO communication
- The frame design and securing components could be enhanced
- Training YOLO for classroom-specific objects and people identification

## Conclusion

The project demonstrated the successful use of YOLOv5 object detection to control a car that could follow a person. While there were some limitations and areas for improvement, the project achieved the intended goals and could serve as a basis for further development in the field of autonomous robotics.

---

*Note: This markdown version is a simplified representation of the original write up, which can be found in the repo.*
