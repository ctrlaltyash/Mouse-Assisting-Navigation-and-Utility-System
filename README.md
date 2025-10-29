# Mouse Assisting Navigation and Utility System - MANUS
Project MANUS is a smart vision-based system designed to detect, track, and interpret human hand movements in real time.


Abstract

Project MANUS is a vision-based system developed to detect, track, and interpret human hand movements in real time. The system employs computer vision and machine learning techniques to recognize gestures and finger positions without the need for physical sensors or wearable devices. By translating hand gestures into computational commands, MANUS provides a natural and intuitive interface for human–computer interaction.

Introduction

The increasing demand for contactless and natural interaction methods has driven research in gesture-based control systems. Project MANUS aims to develop a scalable and efficient framework capable of interpreting hand gestures through live video input. The system focuses on accessibility, adaptability, and real-time responsiveness, making it suitable for applications in robotics, augmented and virtual reality, education, and assistive technologies.

System Architecture

The MANUS system consists of the following major components:
Input Acquisition: Captures live video frames through a standard camera.
Hand Detection: Isolates the hand region using image processing and segmentation algorithms.
Landmark Tracking: Identifies and tracks key hand landmarks to estimate movement and pose.
Gesture Recognition: Classifies gestures based on spatial and temporal patterns using trained models.
Action Mapping: Translates recognized gestures into specific control signals or software actions.
This modular architecture ensures flexibility and allows individual components to be independently updated or replaced.


Technology Stack

Programming Language: Python
Core Libraries: OpenCV, MediaPipe, NumPy
Development Environment: Visual Studio Code / Jupyter Notebook
Hardware Requirements: Standard webcam or compatible camera module


Key Features

Real-time hand and gesture tracking
Non-invasive and sensor-free operation
Modular and extensible architecture
Compatible with multiple domains such as robotics, virtual reality, and assistive systems


Installation
git clone https://github.com/Yash_12711/MANUS.git
cd MANUS
pip install -r requirements.txt
python manus.py


Ensure that the camera device is connected and accessible prior to execution.

Usage
python manus.py --mode gesture

Once executed, the system will open a live video feed. Hand movements performed within the frame will be detected, tracked, and classified in real time.

Potential Applications
Human–computer interaction
Virtual and augmented reality interfaces
Sign language recognition systems
Robotic control modules
Educational and demonstrative platforms


License

This project is licensed under the MIT License
.
You are free to use, modify, and distribute this software, provided proper attribution is given to the original author.

Author

Kalepu Yashvardhan
Project MANUS | 2025
GitHub: Yash_12711
