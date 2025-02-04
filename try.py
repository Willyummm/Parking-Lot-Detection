import cv2
import random
import numpy as np

# Initialize video capture
video = cv2.VideoCapture("BusyParkingLotAerialTimeLapse.mp4")

# Create background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=70)

# Dictionary to store objects and their positions, colors, and movement direction
objects = {}
object_id = 0
distance_threshold = 300

# Define the spots of interest in the frame and initialize them as open
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define target color and tolerance for determining spot occupancy
target_color = (100, 120, 130)  # Example target color
color_tolerance = 30  # Tolerance level for color difference

# Spot definitions
spots = {
    1: {"position": (710, 539), "status": "open", "color": (0, 255, 0)},
    2: {"position": (750, 503), "status": "open", "color": (0, 255, 0)},
    3: {"position": (750, 585), "status": "open", "color": (0, 255, 0)},
    4: {"position": (710, 585), "status": "open", "color": (0, 255, 0)},
    5: {"position": (710, 611), "status": "open", "color": (0, 255, 0)},
    6: {"position": (750, 611), "status": "open", "color": (0, 255, 0)},
}

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Apply background subtraction to detect moving objects
    fg_mask = background_subtractor.apply(frame)

    # Find contours from the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    current_frame_objects = {}
    for box in boxes:
        (x, y, w, h) = box
        center = (x + w // 2, y + h // 2)
        best_match_id = None
        min_distance = float('inf')
        for obj_id, (prev_center, color, direction) in objects.items():
            dist = abs(center[0] - prev_center[0]) + abs(center[1] - prev_center[1])
            if dist < distance_threshold and dist < min_distance:
                best_match_id = obj_id
                min_distance = dist

        if best_match_id is not None:
            prev_center, _, prev_direction = objects[best_match_id]
            direction = 'right' if center[0] > prev_center[0] else 'left'
            color = (0, 0, 255) if direction == 'right' else (255, 0, 0)
            current_frame_objects[best_match_id] = (center, color, direction)
            obj_id = best_match_id
        else:
            color = (0, 255, 0)
            direction = 'stationary'
            current_frame_objects[object_id] = (center, color, direction)
            obj_id = object_id
            object_id += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"ID {obj_id} ({direction})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for spot_id, spot_info in spots.items():
        spot_x, spot_y = spot_info["position"]
        pixel_color = frame[spot_y, spot_x]
        color_difference = np.linalg.norm(pixel_color - target_color)

        if color_difference != color_tolerance:
            spot_info["status"] = "occupied"
            spot_info["color"] = (0, 0, 255)  # Red for occupied
        else:
            spot_info["status"] = "open"
            spot_info["color"] = (0, 255, 0)  # Green for open

    for spot_id, spot_info in spots.items():
        cv2.circle(frame, spot_info["position"], 10, spot_info["color"], -1)
        cv2.putText(frame, f"Spot {spot_id}", (spot_info["position"][0] - 10, spot_info["position"][1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, spot_info["color"], 2)

    cv2.imshow("Parking Lot Status", frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
