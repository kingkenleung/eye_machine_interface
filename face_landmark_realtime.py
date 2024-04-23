import cv2
import mediapipe as mp
import pyautogui
import pprint
import time

# https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
frame_timestamp_ms = 1000
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
screen_size = pyautogui.size()
print(f"Current Screen Size is {screen_size}")

cursor_buffer = []
click_timestamps = {'left': 0, 'right': 0}
click_cooldown = 0.5  # 500 milliseconds cooldown for clicks


# Create a face landmarker instance with the live stream mode:
def cursor_control(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    face_blendshapes = result.face_blendshapes
    target_face_blendshapes = ['eyeBlinkLeft', 'eyeBlinkRight',
                               'eyeLookDownLeft', 'eyeLookDownRight',
                               'eyeLookInLeft', 'eyeLookInRight',
                               'eyeLookOutLeft', 'eyeLookOutRight',
                               'eyeLookUpLeft', 'eyeLookUpRight']
    
    if face_blendshapes:
        eye_coordinates = {category.category_name: category.score for category in face_blendshapes[0] if category.category_name in target_face_blendshapes}
        eye_vectors = {"up": (eye_coordinates['eyeLookUpLeft'] + eye_coordinates['eyeLookUpRight']) / 2,
                       "down": (eye_coordinates['eyeLookDownLeft'] + eye_coordinates['eyeLookDownRight']) / 2,
                       "left": (eye_coordinates['eyeLookOutLeft'] + eye_coordinates['eyeLookInRight']) / 2,
                       "right": (eye_coordinates['eyeLookInLeft'] + eye_coordinates['eyeLookOutRight']) / 2}
        
        try:
            # Correction constant used to counteract position of webcam
            CC = {"left": 2, "right": 1.5, "up": 3, "down": 4}
            
            cursor_x = (1 - eye_vectors['left'] * CC['left'] + eye_vectors['right'] * CC['right']) * screen_size[0] / 2
            cursor_y = (1 - eye_vectors['up'] * CC['up'] + eye_vectors['down'] * CC['down']) * screen_size[1] / 2

            # Add to buffer and calculate average position
            cursor_buffer.append((cursor_x, cursor_y))
            if len(cursor_buffer) > 5:  # Keep only the last 5 positions
                cursor_buffer.pop(0)
            
            avg_x = sum(pos[0] for pos in cursor_buffer) / len(cursor_buffer)
            avg_y = sum(pos[1] for pos in cursor_buffer) / len(cursor_buffer)

            # Move cursor to the averaged position
            pyautogui.moveTo(avg_x, avg_y)
            
            
            # Initialize flags for eye blinks
            is_eye_blink_left = False
            is_eye_blink_right = False
            
            # Define a threshold for detecting a blink
            blink_threshold = 0.4

            # Check each category to see if it's a blink and if the score is above the threshold
            for category in face_blendshapes[0]:
                if category.category_name == 'eyeBlinkLeft' and category.score > blink_threshold:
                    if current_time - click_timestamps['left'] > click_cooldown:
                        is_eye_blink_left = True
                        print(f"Eye Blink Left: {is_eye_blink_left} ({category.score})")
                        pyautogui.click(button='left')
                        click_timestamps['left'] = current_time
                elif category.category_name == 'eyeBlinkRight' and category.score > blink_threshold:
                    if current_time - click_timestamps['right'] > click_cooldown:
                        is_eye_blink_right = True
                        print(f"Eye Blink Right: {is_eye_blink_right} ({category.score})")
                        pyautogui.click(button='right')
                        click_timestamps['right'] = current_time
        except:
            print("Something went wrong!")


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=cursor_control,
    num_faces=1)

# Initialize the webcam.
cap = cv2.VideoCapture(0)

while True:

    with FaceLandmarker.create_from_options(options) as landmarker:
        ret, frame = cap.read()
        if not ret:
            break
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, frame_timestamp_ms)


        # Break the loop when 'ESC' is pressed.
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the webcam and close the window.
cap.release()
# cv2.destroyAllWindows()
