import cv2
import numpy as np
import datetime

# Global variables to store initial quadrant positions
initial_quadrants = []

def detect_initial_quadrants(frame):
    global initial_quadrants

    # Convert frame to grayscale for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate lines
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the vertices of detected quadrants
    quadrants = []

    # Loop through contours and identify potential quadrants
    for cnt in contours:
        # Approximate the contour with a polygon
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon has 4 sides (quadrilateral)
        if len(approx) == 4:
            # Calculate the area of the contour
            area = cv2.contourArea(approx)

            # Discard contours with small areas (noise reduction)
            if area > 1000:  # Adjust threshold as needed
                # Store the vertices of the quadrants
                quadrants.append(approx.reshape(-1, 2))

    # Store initial quadrant positions globally
    initial_quadrants = quadrants

    return quadrants

def detect_hand_placed_objects(frame, quadrants):
    global initial_quadrants

    # Convert frame to grayscale for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate hand-placed objects
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected objects within quadrants
    detected_objects = []

    # Loop through contours and detect objects within defined quadrants
    for contour in contours:
        # Calculate bounding rectangle around contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate center of bounding rectangle
        cX = x + w // 2
        cY = y + h // 2

        # Determine which quadrant the object is in based on initial quadrant positions
        for quad_num, quad in enumerate(initial_quadrants, start=1):
            # Convert quadrant to polygon
            polygon = np.array(quad, dtype=np.int32)
            
            # Check if any part of the object contour is within the quadrant area
            if cv2.pointPolygonTest(polygon, (cX, cY), False) >= 0:
                # Detect color of the object within the contour
                color = detect_color(frame[y:y+h, x:x+w])
                
                # Append detected object with color and quadrant information
                detected_objects.append((cX, cY, color, quad_num))
                break

    return detected_objects

def detect_color(region):
    # Define HSV ranges for different colored objects
    color_ranges = {
        'green': (np.array([35, 50, 50]), np.array([90, 255, 255])),
        'yellow': (np.array([20, 100, 100]), np.array([30, 255, 255])),
        'red': (np.array([0, 100, 100]), np.array([10, 255, 255])),
        'white': (np.array([0, 0, 200]), np.array([180, 50, 255])),
        'orange': (np.array([5, 100, 100]), np.array([15, 255, 255]))
    }

    # Convert region to HSV
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Initialize variables to store color detection results
    detected_color = "Unknown"
    max_area = 0

    # Loop through each color range and detect color within the region
    for color, (lower, upper) in color_ranges.items():
        # Threshold the HSV image to get only color
        mask_color = cv2.inRange(hsv, lower, upper)

        # Find contours in the color mask
        contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour is found
        if contours:
            # Calculate area of largest contour
            area = max(cv2.contourArea(cnt) for cnt in contours)

            # Update detected color if current area is larger
            if area > max_area:
                detected_color = color
                max_area = area

    return detected_color

# Open the video file
cap = cv2.VideoCapture("AI Assignment video.mp4")  # Replace with your video path

# Detect initial quadrants
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        break

    # Detect initial quadrants and exit loop
    initial_quadrants = detect_initial_quadrants(frame)

    break

# Reset video capture to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        break

    # Detect hand-placed objects within each quadrant
    objects = detect_hand_placed_objects(frame.copy(), initial_quadrants)

    # Draw detected quadrants on the frame (optional)
    for quad in initial_quadrants:
        cv2.polylines(frame, [quad], True, (0, 255, 0), 2)

    # Draw detected objects and their respective quadrant info on the frame
    for obj in objects:
        cX, cY, color, quad_num = obj
        cv2.circle(frame, (cX, cY), 10, (0, 0, 255), -1)
        cv2.putText(frame, f'{color} Object in Quadrant {quad_num}', (cX - 10, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Get current timestamp
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Print time, color, and quadrant information
        print(f"Time: {current_time}, {color} Object detected in Quadrant {quad_num}")

    # Display the resulting frame
    cv2.imshow("Quadrant and Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
