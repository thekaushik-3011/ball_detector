import cv2
import numpy as np
import datetime

initial_quadrants = []
tracked_objects = {}
events = []
quadrant_names = {
    1: 'Quadrant 4',
    2: 'Quadrant 1',
    3: 'Quadrant 2',
    4: 'Quadrant 3'
}

def detect_initial_quadrants(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrants = []

    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 1000:
                quadrants.append(approx.reshape(-1, 2))

    return quadrants

def detect_balls(frame, quadrants):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'orange': ([10, 100, 100], [20, 255, 255]),
        'white': ([0, 0, 200], [180, 55, 255])
    }

    detected_balls = []

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                continue

            cX = x + w // 2
            cY = y + h // 2

            for quad_num, quad in enumerate(quadrants, start=1):
                polygon = np.array(quad, dtype=np.int32)

                if cv2.pointPolygonTest(polygon, (cX, cY), False) >= 0:
                    detected_balls.append((x, y, w, h, cX, cY, quadrant_names[quad_num], color))
                    break

    return detected_balls

def update_tracked_objects(detected_balls, current_time):
    global tracked_objects, events

    current_objects = {}

    for ball in detected_balls:
        x, y, w, h, cX, cY, quad_name, color = ball

        obj_id = (cX, cY, color)
        if obj_id not in tracked_objects:
            tracked_objects[obj_id] = {'quadrant': quad_name, 'status': 'entry'}
            events.append((current_time, quad_name, color, 'Entry'))
            print(f"Time: {current_time}, Object detected in {quad_name}, Color: {color}, Event: Entry")
        elif tracked_objects[obj_id]['quadrant'] != quad_name:
            tracked_objects[obj_id]['quadrant'] = quad_name
            events.append((current_time, quad_name, color, 'Entry'))
            print(f"Time: {current_time}, Object detected in {quad_name}, Color: {color}, Event: Entry")

        current_objects[obj_id] = tracked_objects.pop(obj_id)

    for obj_id, obj_info in tracked_objects.items():
        events.append((current_time, obj_info['quadrant'], obj_id[2], 'Exit'))
        print(f"Time: {current_time}, Object exited from {obj_info['quadrant']}, Color: {obj_id[2]}, Event: Exit")

    tracked_objects = current_objects

def save_events_to_file(events, filename="events.txt"):
    with open(filename, 'w') as file:
        for event in events:
            file.write(f"{event[0]}, {event[1]}, {event[2]}, {event[3]}\n")

cap = cv2.VideoCapture("AI Assignment video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    initial_quadrants = detect_initial_quadrants(frame)
    break

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
screen_width = 1280
screen_height = 720

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.timedelta(seconds=int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
    balls = detect_balls(frame.copy(), initial_quadrants)
    update_tracked_objects(balls, current_time)

    for quad in initial_quadrants:
        cv2.polylines(frame, [quad], True, (0, 255, 0), 2)

    for ball in balls:
        x, y, w, h, cX, cY, quad_name, color = ball
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'Ball in {quad_name} - {color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    frame_resized = cv2.resize(frame, (screen_width, screen_height))
    out.write(frame_resized)
    cv2.imshow("Quadrant and Ball Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

save_events_to_file(events)
cap.release()
out.release()
cv2.destroyAllWindows()
