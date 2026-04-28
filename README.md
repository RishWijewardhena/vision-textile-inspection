# Vision Textile Inspection

A computer vision-based fabric inspection system using deep learning for automated defect detection in textiles.

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements an automated textile quality inspection system using YOLOv8 segmentation models. The system can detect, classify, and measure fabric defects in real-time, providing accurate quality control for textile manufacturing processes. It includes hardware integration for serial communication, MQTT connectivity, and automated measurement capabilities.

### Key Features

- Real-time fabric defect detection using YOLOv8 segmentation
- Dual model support for defect and single-needle detection
- Advanced calibration system for accurate dimensional measurements
- Database integration for defect tracking and analysis
- Serial communication interface for hardware control
- MQTT connectivity for remote monitoring and heartbeat
- Automated image capture and analysis
- Comprehensive measurement and stitch analysis
- Support for multiple defect types and classifications
- Automated annotation saving for quality records
- Logger system for monitoring and troubleshooting

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/RishWijewardhena/vision-textile-inspection.git
   cd vision-textile-inspection
   ```

2. **Create a virtual environment**
   
   For Linux/macOS:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   For Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the root directory with your configuration settings (if needed).

## Usage

### Running the Main Inspection System

Start the fabric inspection application:

```bash
python main.py
```

### Running the Calibration Tool

Before performing inspections, calibrate the system for accurate measurements:

```bash
python calibration.py
```

### Database Management

Access database operations directly:

```bash
python database.py
```

### Measurement Analysis

Perform measurement calculations and analysis:

```bash
python measurement.py
```

### Automated Execution

Run automated inspection scripts:

```bash
bash auto_run.sh
```

### Utility Scripts

Access various utility scripts in the `Utils/` folder:
- `auto_capture.py`: Automated image capture
- `check_model.py`: Model validation and testing
- `check_stitch_distance.py`: Stitch distance analysis
- `usb_camera.py`: USB camera interface
- `mqtt_reset_test.py`: MQTT reset testing
- `test_reset.py`: System reset testing

## Project Structure

```
THREAD/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration settings
├── main.py                        # Main inspection application
├── calibration.py                 # Camera calibration module
├── measurement.py                 # Measurement and analysis utilities
├── database.py                    # Database operations
├── hardware_utils.py              # Hardware interface utilities
├── serial_reader.py               # Serial communication module
├── mqtt_heartbeat.py              # MQTT heartbeat/connectivity module
├── file_cleaner.py                # File management utilities
├── best_Model.pt                  # YOLOv8 model for defect detection (angled camera)
├── single_needle_model.pt         # YOLOv8 model for single needle detection
├── camera_calibration.json        # Camera calibration data
├── camera_extrinsics.json         # Camera extrinsics data
├── ChArUco_Calibration_Linux      # ChArUco calibration application
├── auto_run.sh                    # Automated run script
├── auto_runner.sh                 # Alternative run script
├── download_calibartion_app.sh    # Calibration app downloader
├── .gitignore                     # Git ignore rules
├── .env                           # Environment variables
├── __pycache__/                   # Python cache (ignored)
├── venv/                          # Virtual environment (ignored)
├── logs/                          # Application logs
├── saved_annotations/             # Saved annotations and results
│   ├── 2026-04-27_01-01-13/
│   ├── 2026-04-27_15-21-43/
│   ├── 2026-04-28_06-00-33/
│   ├── 2026-04-28_08-28-41/
│   └── 2026-04-28_08-30-30/
├── Utils/                         # Utility scripts and tools
│   ├── auto_capture.py
│   ├── check_model.py
│   ├── check_stitch_distance.py
│   ├── mqtt_reset_test.py
│   ├── test_reset.py
│   ├── usb_camera.py
│   ├── single_needle_model.pt
│   ├── calibration_app_link.txt
│   └── single_needle_photos_np02.zip
└── Testing/                       # Test scripts
    └── test1.py
```

### Module Descriptions

- **main.py**: Core inspection logic and defect detection pipeline
- **calibration.py**: Handles camera calibration for accurate spatial measurements
- **measurement.py**: Measurement calculations and analysis utilities
- **database.py**: Database connectivity and data storage operations
- **config.py**: Configuration settings and parameters
- **hardware_utils.py**: Hardware interface and utilities
- **serial_reader.py**: Serial communication interface for hardware devices
- **mqtt_heartbeat.py**: MQTT connectivity and heartbeat monitoring
- **file_cleaner.py**: File management and cleanup utilities
- **best_Model.pt**: YOLOv8 segmentation model trained for defect detection (angled camera mount)
- **single_needle_model.pt**: YOLOv8 segmentation model for single needle detection

## Configuration

The system can be configured through environment variables or configuration files. Key parameters include:

- Camera resolution and frame rate
- Detection confidence threshold
- Model input size
- Database connection settings
- Calibration parameters

## Dependencies

### Core Libraries

- **Python**: 3.11+
- **OpenCV**: Computer vision operations and image processing
- **NumPy**: Numerical computations and array operations
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLOv8 implementation

### Complete Dependency List

See `requirements.txt` for the full list of dependencies with version specifications.

## Model Information

The project uses a YOLOv8 medium segmentation model (`best_Model.pt`) trained specifically for textile defect detection. The model can identify and segment various types of fabric defects including:

- Holes and tears
- Stains and discoloration
- Thread irregularities
- Pattern defects
- Other manufacturing flaws

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Rish Wijewardhena**

- GitHub: [@RishWijewardhena](https://github.com/RishWijewardhena)
- Project Link: [https://github.com/RishWijewardhena/vision-textile-inspection](https://github.com/RishWijewardhena/vision-textile-inspection)

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch team

---

**Note**: This project is under active development. Features and documentation may be updated regularly.