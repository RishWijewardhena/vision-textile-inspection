# Vision Textile Inspection

A Python project for fabric inspection using computer vision and deep learning.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Contact](#contact)

## Description
This project uses computer vision techniques to detect defects in fabric using YOLO segmentation models. It supports calibration, measurement, and data storage for textile quality inspection.

## Installation
1. Clone the repository:
```bash
git clone https://github.com/RishWijewardhena/vision-textile-inspection.git

python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  

pip install -r requirements.txt

Run calibration App
Main_code/
├── .gitignore
├── README.md
├── calibration.py
├── main.py
├── database.py
├── yolov8n_seg_200.pt
├── __pycache__/      # ignored
├── .env/             # ignored
├── saved_annotations/ # ignored

Dependencies

Python 3.11+
OpenCV
Numpy
PyTorch
Ultralytics YOLO
