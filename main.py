from ultralytics import YOLO
import sort
import cv2
import cvzone
import math
import numpy as np

# Load the video
cap = cv2.VideoCapture('media/cars.mp4')

# Yolo model
model = YOLO('models/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Mask
mask = cv2.imread("media/mask.png")
mask = cv2.resize(mask, (854, 480), interpolation=cv2.INTER_NEAREST)

# Tracking
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Coordinates for the line
limits = [0, 330, 853, 330]

# Vehicle counter variable
totalCount = set()

while True:
    # Reading the video capture
    success, img = cap.read()

    # Aplying the mask to the video using bitwise pixel by pixel
    imgRegion = cv2.bitwise_and(img, mask)

    # Run the model on the image region with streaming enabled on GPU (device 0)
    results = model(imgRegion, stream=True, device=0)

    # Numpy array for tracking. In the format: [x1,y1,x2,y2,score]
    detections = np.empty((0, 5))

    # Iterate over the results
    for r in results:
        boxes = r.boxes

        # For each box in the frame/image
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)            
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # If the object is a vehicle
            if(currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3):
                # Fill a numpy array and populate the detections array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Tracker results
    resultsTracker = tracker.update(detections)

    # Draw the line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        # Get the coodinates of result
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        w, h = x2 - x1, y2 - y1

        # Draw the box and the image using cvzone
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3)
        cvzone.cornerRect(img, (x1,y1,w,h), rt=1, t=2, l=9, colorR=(255, 0, 255))

        # Calculation and draw of box's centers
        cx = x1 + w//2
        cy = y1 + h//2
        cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        # If the centers cross the line
        if(limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15):
            totalCount.add(Id)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # Turn line green

    # Put a text box of the counter
    cvzone.putTextRect(img, f'Total Vehicles: {len(totalCount)}',
                       (40, 40), 
                       scale=2, thickness=2, offset=10, 
                       colorR=(255, 0, 255))
        

    cv2.imshow("Image", img)
    cv2.waitKey(1)
