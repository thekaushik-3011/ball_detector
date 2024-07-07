import cv2
import numpy as np
import datetime

# Global variables to store initial quadrant positions and track events
initial_quadrants = []
tracked_objects = {}
events = []

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

def detect_balls(frame, quadrants):
    global initial_quadrants

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges for the balls
    color_ranges = {
        'green': ([35, 50, 50], [90, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'orange': ([10, 100, 100], [20, 255, 255]),
        'white': ([0, 0, 200], [180, 55, 255])
    }

    # List to store detected balls within quadrants
    detected_balls = []

    # Loop through each color range and detect balls
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # Filter out small contours (noise reduction)
            if area < 1000:  # Adjust threshold as needed
                continue

            # Calculate bounding rectangle around contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate center of bounding rectangle
            cX = x + w // 2
            cY = y + h // 2

            # Determine which quadrant the ball is in based on initial quadrant positions
            for quad_num, quad in enumerate(initial_quadrants, start=1):
                # Convert quadrant to polygon
                polygon = np.array(quad, dtype=np.int32)

                # Check if the ball's center is within the quadrant
                if cv2.pointPolygonTest(polygon, (cX, cY), False) >= 0:
                    detected_balls.append((x, y, w, h, cX, cY, quad_num, color))
                    break

    return detected_balls

def update_tracked_objects(frame, detected_balls, current_time):
    global tracked_objects, events

    for ball in detected_balls:
        x, y, w, h, cX, cY, quad_num, color = ball

        obj_id = (cX, cY, color)
        if obj_id not in tracked_objects:
            tracked_objects[obj_id] = {'quadrant': quad_num, 'status': 'entry'}
            events.append((current_time, quad_num, color, 'Entry'))
        elif tracked_objects[obj_id]['quadrant'] != quad_num:
            tracked_objects[obj_id]['quadrant'] = quad_num
            events.append((current_time, quad_num, color, 'Entry'))

    for obj_id in list(tracked_objects.keys()):
        if obj_id not in [(cX, cY, color) for x, y, w, h, cX, cY, quad_num, color in detected_balls]:
            events.append((current_time, tracked_objects[obj_id]['quadrant'], obj_id[2], 'Exit'))
            del tracked_objects[obj_id]

def save_events_to_file(events, filename="events.txt"):
    with open(filename, 'w') as file:
        for event in events:
            file.write(f"{event[0]}, {event[1]}, {event[2]}, {event[3]}\n")

# Open the video file
cap = cv2.VideoCapture("AI Assignment video.mp4")  # Replace with your video path

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps

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

# Define the codec and create VideoWriter object for mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Get screen size
screen_width = 1920  # Adjust based on your screen resolution
screen_height = 1080  # Adjust based on your screen resolution

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # If frame is not read successfully, break the loop
    if not ret:
        break

    # Get current time based on video duration
    current_time = datetime.timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))

    # Detect balls within each quadrant
    balls = detect_balls(frame.copy(), initial_quadrants)

    # Update tracked objects and record events
    update_tracked_objects(frame, balls, current_time)

    # Draw detected quadrants on the frame (optional)
    for quad in initial_quadrants:
        cv2.polylines(frame, [quad], True, (0, 255, 0), 2)

    # Draw detected balls and their respective quadrant info on the frame
    for ball in balls:
        x, y, w, h, cX, cY, quad_num, color = ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'Ball in Quadrant {quad_num} - {color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Resize the frame to fit the screen
    frame_resized = cv2.resize(frame, (screen_width, screen_height))

    # Write the frame to the output video
    out.write(frame_resized)

    # Display the resulting frame
    cv2.imshow("Quadrant and Ball Detection", frame_resized)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save events to file
save_events_to_file(events)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
