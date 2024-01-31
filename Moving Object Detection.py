import cv2

cap = cv2.VideoCapture(0)

# Set the initial frame as None
prev_frame = None

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the frame to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the previous frame is None, initialize it
    if prev_frame is None:
        prev_frame = gray
        continue

    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(prev_frame, gray)

    # Apply a threshold to the frame difference to convert it to a binary image
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in the holes
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there are contours in the thresholded image, the object is moving
    if len(contours) > 0:
        print("Object is moving")
    else:
        print("Object is not moving")

    # Display the current frame
    cv2.imshow("Frame", frame)

    # Update the previous frame
    prev_frame = gray

    # Check for quit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()
