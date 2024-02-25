import cv2

from lib.camera import OAKDCamera, ImageCamera, VideoCamera
from lib.hsv_filter import detectLine
# from lib.actuator import VESC
       
IMAGE_TO_TEST = "test_images/f1-test.png"

camera = ImageCamera(path=IMAGE_TO_TEST)

# VIDEO_TO_TEST = "test_images/f110_fpv.mp4"
# camera = VideoCamera(path=VIDEO_TO_TEST)
# vehicle = VESC()
detector = detectLine(camera)
while True:
    detector.run()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("--- Saving Settings ---")
        detector.save_settings() # Choose to save settings if in calibration mode
        if detector.calibration_mode:
            cv2.destroyAllWindows()
        break
    
    