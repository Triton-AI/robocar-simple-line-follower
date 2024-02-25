import cv2
import depthai as dai
import imutils
from .hsv_filter import detectLine

class BaseCamera():
    def __init__(self, img):
        self.img = img
    def get_frame(self):
        return self.img
    def show_frame():
        pass
    def __del__():
        pass

class VideoCamera(BaseCamera):
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            raise ValueError("Unable to open video file: {}".format(path))
        self.line_detector = detectLine(self)  
        
    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None
        return frame

    def show_frame(self):
        ret, frame = self.video.read()
        if ret:
            annotated_frame = self.annotate_frame(frame)
            cv2.imshow('frame', annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                return False
            return True
        else:
            return False
    
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()


class ImageCamera(BaseCamera):
    def __init__(self, path=None, resolution=(300, 300)):
        # load image from path
        self.image = cv2.imread(path)
        self.image = cv2.resize(self.image, resolution)

    def show_frame(self):
        cv2.imshow('frame', self.image)
        
    def get_frame(self):
        return self.image
    
    def __del__(self):
        cv2.destroyAllWindows()

class OAKDCamera(BaseCamera):
    def __init__(self, resolution=(300,300)) -> None:
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb.setStreamName("rgb")
        self.camRgb.setPreviewSize(resolution[0], resolution[1])
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.camRgb.preview.link(self.xoutRgb.input)
        self.device = dai.Device(self.pipeline)
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.frame = None

    def get_frame(self):
        inRgb = self.qRgb.get()
        self.frame = inRgb.getCvFrame()
        return self.frame
    
    def show_frame(self):
        cv2.imshow("rgb", self.frame)
        if cv2.waitKey(1) == ord('q'):
            return False
        return True
    
    def __del__(self):
        cv2.destroyAllWindows()
        self.device.close()
