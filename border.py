import cv2

def detect_quadrants(frame):
  # Convert frame to grayscale for better processing
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Threshold the image to isolate lines
  thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

  # Find contours in the thresholded image
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Loop through contours and identify potential quadrants
  for cnt in contours:
    # Approximate the contour with a polygon
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    # Check if the polygon has 4 sides (quadrilateral)
    if len(approx) == 4:
      # Calculate the center of the contour
      M = cv2.moments(cnt)
      if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
      else:
        cX = 0
        cY = 0

      # Check if lines intersect near the center (assuming quadrant lines are thick)
      intersection_count = 0
      for p in approx:
        px, py = p[0]
        # Check for white pixels (indicating line intersection) in a small area around the center
        mask = thresh[py-5:py+5, px-5:px+5]
        intersection_count += cv2.countNonZero(mask)

      # If there are enough intersections, consider it a quadrant
      if intersection_count > 20:
        # Draw a bounding box around the detected quadrant
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  # Return the frame with potential quadrants highlighted
  return frame

# Open the video file
cap = cv2.VideoCapture("AI Assignment video.mp4")  # Replace with your video path

while True:
  # Read a frame from the video
  ret, frame = cap.read()

  # If frame is not read successfully, break the loop
  if not ret:
    break

  # Detect quadrants in the frame
  frame_with_quadrants = detect_quadrants(frame.copy())

  # Display the resulting frame
  cv2.imshow("Quadrant Detection", frame_with_quadrants)

  # Exit on 'q' key press
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()
