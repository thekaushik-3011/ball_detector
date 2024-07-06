import cv2
import numpy as np
import time

video_path = 'AI Assignment video.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the processed video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('processed_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Define the bounding box for the quadrants
quadrant_x1, quadrant_y1 = 100, 100  # Top-left corner of the red-bordered area
quadrant_x2, quadrant_y2 = 500, 500  # Bottom-right corner of the red-bordered area

# Define the quadrants within the bounding box
def get_quadrant(x, y):
    if x < (quadrant_x1 + quadrant_x2) / 2 and y < (quadrant_y1 + quadrant_y2) / 2:
        return 3
    elif x >= (quadrant_x1 + quadrant_x2) / 2 and y < (quadrant_y1 + quadrant_y2) / 2:
        return 4
    elif x < (quadrant_x1 + quadrant_x2) / 2 and y >= (quadrant_y1 + quadrant_y2) / 2:
        return 2
    else:
        return 1

# Color bounds for different balls
colors = ['green', 'orange', 'white', 'yellow']
lower_bounds = [
    (29, 86, 6),  # Lower bound for green
    (0, 100, 100),  # Lower bound for orange
    (0, 0, 200),  # Lower bound for white
    (22, 93, 0)  # Lower bound for yellow
]
upper_bounds = [
    (64, 255, 255),  # Upper bound for green
    (20, 255, 255),  # Upper bound for orange
    (180, 255, 255),  # Upper bound for white
    (45, 255, 255)  # Upper bound for yellow
]

prev_positions = {color: (0, 0) for color in colors}

def detect_balls(frame, lower_bounds, upper_bounds):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_positions = {}

    for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Filter by area
                moments = cv2.moments(largest_contour)
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                ball_positions[colors[i]] = (cx, cy)
    return ball_positions

ball_positions = {}
events = []

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        detected_balls = detect_balls(frame, lower_bounds, upper_bounds)

        for ball_color, (x, y) in detected_balls.items():
            if quadrant_x1 <= x <= quadrant_x2 and quadrant_y1 <= y <= quadrant_y2:
                quadrant = get_quadrant(x, y)

                if ball_color in ball_positions:
                    prev_x, prev_y = ball_positions[ball_color]
                    prev_quadrant = get_quadrant(prev_x, prev_y)
                    if quadrant != prev_quadrant:
                        event_type = 'Exit' if prev_quadrant != quadrant else 'Entry'
                        timestamp = time.time() - start_time
                        events.append((timestamp, prev_quadrant, ball_color, 'Exit'))
                        events.append((timestamp, quadrant, ball_color, 'Entry'))
                        cv2.putText(frame, f'{event_type} {quadrant}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                ball_positions[ball_color] = (x, y)

        out.write(frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save the events to a text file
with open('events.txt', 'w') as f:
    for event in events:
        f.write(','.join(map(str, event)) + '\n')
