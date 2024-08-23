# Vehicle Counter Using YOLO and SORT

This project is designed to count vehicles in a video stream using the YOLOv8 model for object detection and SORT for tracking. It processes video frames to detect and track vehicles, such as cars, trucks, buses, and motorbikes, and counts them when they cross a predefined line in the video.

## Installation

To run this project, make sure you have Python installed. Then, install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/seu-usuario/Vehicle-Counter-YOLO.git
   cd Vehicle-Counter-YOLO
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your video file in the `media/` directory.

4. **Modify the script:**
   - Open the `main.py` script.
   - Change the path to your video file on line `# Load the video`:
     ```python
     cap = cv2.VideoCapture('media/your_video.mp4')
     ```
   - Adjust the coordinates of the counting line on line `# Coordinates for the line` to fit your video:
     ```python
     limits = [x1, y1, x2, y2]
     ```
     Here, `x1, y1` represent the starting point, and `x2, y2` represent the endpoint of the line.

5. Run the script:
   ```bash
   python main.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [SORT](https://github.com/abewley/sort)
- [cvzone](https://github.com/cvzone/cvzone)
