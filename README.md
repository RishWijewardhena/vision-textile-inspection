# Vision Textile Inspection

A computer vision-based fabric inspection system using deep learning for automated defect detection in textiles.

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements an automated textile quality inspection system using YOLOv8 segmentation models. The system can detect, classify, and measure fabric defects in real-time, providing accurate quality control for textile manufacturing processes.

### Key Features

- Real-time fabric defect detection using YOLOv8 segmentation
- Calibration system for accurate dimensional measurements
- Database integration for defect tracking and analysis
- Support for multiple defect types and classifications
- Automated annotation saving for quality records

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

### Running the Calibration Tool

Before performing inspections, calibrate the system for accurate measurements:

```bash
python calibration.py
```

### Running the Main Inspection System

Start the fabric inspection application:

```bash
python main.py
```

### Database Management

Access database operations directly:

```bash
python database.py
```

## Project Structure

```
Main_code/
├── .gitignore                # Git ignore rules
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── calibration.py           # Camera calibration module
├── main.py                  # Main inspection application
├── database.py              # Database operations
├── yolov8n_seg_200.pt      # Pre-trained YOLO model Old
├── best_Model.pt           #re trained  model for angled camera mount 
├── __pycache__/             # Python cache (ignored)
├── .env/                    # Virtual environment (ignored)
└── saved_annotations/       # Annotation storage (ignored)
```

### Module Descriptions

- **calibration.py**: Handles camera calibration for accurate spatial measurements
- **main.py**: Core inspection logic and defect detection pipeline
- **database.py**: Database connectivity and data storage operations
- **best_Model.pt**: YOLOv8 nano segmentation model trained on textile defects

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