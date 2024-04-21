import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 1

# https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
frame_timestamp_ms = 1000
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    face_blendshapes = result.face_blendshapes
    target_face_blendshapes = ['eyeBlinkLeft', 'eyeBlinkRight']
    
    # Initialize flags for eye blinks
    is_eye_blink_left = False
    is_eye_blink_right = False
    
    # Define a threshold for detecting a blink
    blink_threshold = 0.4
    
    if face_blendshapes:
#         filtered_blendshapes = [category for category in face_blendshapes[0] if category.category_name in target_face_blendshapes]
#         print(filtered_blendshapes)
        
        # Check each category to see if it's a blink and if the score is above the threshold
        for category in face_blendshapes[0]:
            if category.category_name == 'eyeBlinkLeft' and category.score > blink_threshold:
                is_eye_blink_left = True
                print(f"Eye Blink Left: {is_eye_blink_left} ({category.score})")
                pyautogui.click(button='left')
            elif category.category_name == 'eyeBlinkRight' and category.score > blink_threshold:
                is_eye_blink_right = True
                print(f"Eye Blink Right: {is_eye_blink_right} ({category.score})")
                pyautogui.click(button='right')


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    output_face_blendshapes=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
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
