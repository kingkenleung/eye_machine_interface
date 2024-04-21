import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Landmarker.
mp_face_landmark = mp.solutions.face_mesh
face_mesh = mp_face_landmark.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize the webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

def get_iris_center(landmarks, iris_indices):
    iris_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in iris_indices])
    center = np.mean(iris_points, axis=0)
    return center

def eye_direction(iris_center, upper_eyelid_point, lower_eyelid_point):
    # Calculate the direction vector of the eyelid line
    direction_vector = np.array(lower_eyelid_point) - np.array(upper_eyelid_point)
    # Calculate the vector from the upper eyelid to the iris center
    iris_vector = np.array(iris_center) - np.array(upper_eyelid_point)
    # Calculate the cross product of the vectors
    cross_product = np.cross(direction_vector, iris_vector)
    # Determine the direction based on the z-component of the cross product
    if cross_product > 0:
        return "Looking Down"
    else:
        return "Looking Up"

# Indices for the iris and eyelids.
left_iris_indices = [474, 475, 476, 477]  # Approximate center of the left iris
right_iris_indices = [469, 470, 471, 472]  # Approximate center of the right iris
left_upper_eyelid_idx = 386  # A point on the left upper eyelid
left_lower_eyelid_idx = 374  # A point on the left lower eyelid
right_upper_eyelid_idx = 159  # A point on the right upper eyelid
right_lower_eyelid_idx = 145  # A point on the right lower eyelid

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the face.
    results = face_mesh.process(image)

    # Convert the image back to BGR for displaying.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the face landmarks on the image.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_landmark.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())

            # Get iris centers.
            left_iris_center = get_iris_center(face_landmarks, left_iris_indices)
            right_iris_center = get_iris_center(face_landmarks, right_iris_indices)

            # Get eyelid points.
            left_upper_eyelid_point = (face_landmarks.landmark[left_upper_eyelid_idx].x, face_landmarks.landmark[left_upper_eyelid_idx].y)
            left_lower_eyelid_point = (face_landmarks.landmark[left_lower_eyelid_idx].x, face_landmarks.landmark[left_lower_eyelid_idx].y)
            right_upper_eyelid_point = (face_landmarks.landmark[right_upper_eyelid_idx].x, face_landmarks.landmark[right_upper_eyelid_idx].y)
            right_lower_eyelid_point = (face_landmarks.landmark[right_lower_eyelid_idx].x, face_landmarks.landmark[right_lower_eyelid_idx].y)

            # Determine eye direction.
            left_eye_direction = eye_direction(left_iris_center, left_upper_eyelid_point, left_lower_eyelid_point)
            right_eye_direction = eye_direction(right_iris_center, right_upper_eyelid_point, right_lower_eyelid_point)

            print("Left Eye:", left_eye_direction)
            print("Right Eye:", right_eye_direction)

    # Display the image.
    cv2.imshow('Facial Landmarks', image)

    # Break the loop when 'ESC' is pressed.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window.
cap.release()
cv2.destroyAllWindows()
