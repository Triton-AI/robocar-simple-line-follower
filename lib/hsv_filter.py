from lib.utils import JSONManager
import cv2
import numpy as np
import imutils


class detectLine(JSONManager):
    # Helper functions
    def update_lower_hue(self, value): self.lower_hue = value
    def update_lower_sat(self, value): self.lower_sat = value
    def update_lower_val(self, value): self.lower_val = value
    def update_upper_hue(self, value): self.upper_hue = value
    def update_upper_sat(self, value): self.upper_sat = value
    def update_upper_val(self, value): self.upper_val = value
    def update_left_crop(self, value): self.left_crop = value/100
    def update_right_crop(self, value): self.right_crop = value/100
    def update_top_crop(self, value): self.top_crop = value/100
    def update_bottom_crop(self, value): self.bottom_crop = value/100
    
    def __init__(self, camera, windowName='Camera'):
        self.camera = camera
        self.windowName = windowName
        self.maskWindowName = 'Mask'
        self.cX, self.cY = 0, 0
        self.frame = self._capture_frame()
        self.steering, self.throttle = 0, 0
        super().__init__()
        self._initialize_calibration()


    def _initialize_calibration(self):

        if self.calibration_mode:
            # Create named windows
            cv2.namedWindow(self.windowName)
            cv2.namedWindow(self.maskWindowName)

            # Create trackbars
            cv2.createTrackbar('Lower Hue', 'Mask', self.lower_hue, 254, self.update_lower_hue)
            cv2.createTrackbar('Lower Sat', 'Mask', self.lower_sat, 254, self.update_lower_sat)
            cv2.createTrackbar('Lower Val', 'Mask', self.lower_val, 254, self.update_lower_val)
            cv2.createTrackbar('Upper Hue', 'Mask', self.upper_hue, 255, self.update_upper_hue)
            cv2.createTrackbar('Upper Sat', 'Mask', self.upper_sat, 255, self.update_upper_sat)
            cv2.createTrackbar('Upper Val', 'Mask', self.upper_val, 255, self.update_upper_val)

            # Create buttons
            cv2.createTrackbar('Invert', 'Mask', self.invert_mask, 1, lambda x: setattr(self, 'invert_mask', x))
            cv2.createTrackbar('Run Motor', 'Mask', self.run_motor, 1, lambda x: setattr(self, 'run_motor', x))

            # Create trackbars for region of interest
            cv2.createTrackbar('Left Crop', 'Mask', int(self.left_crop*100), 100, self.update_left_crop)
            cv2.createTrackbar('Right Crop', 'Mask', int(self.right_crop*100), 100, self.update_right_crop)
            cv2.createTrackbar('Top Crop', 'Mask', int(self.top_crop*100), 100, self.update_top_crop)
            cv2.createTrackbar('Bottom Crop', 'Mask', int(self.bottom_crop*100), 100, self.update_bottom_crop)

    def _capture_frame(self):
        return np.copy(self.camera.get_frame())

    def _find_contours(self, mask):
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)
    
    def _calculate_centroid(self, mask):
        M = cv2.moments(mask)
        print(M["m00"])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        return None, None

    def _calculate_actuator_values(self, cX, cY):
        if cX is not None and cY is not None:
            # Limit the steering angle from 0 to 1
            self.steering = max(min(1-(self.frame.shape[1]-self.cX)/(self.frame.shape[1]),1),0)

            if self.run_motor:
                # Clamp the steering angle from 0 to 1
                if self.steering > 0.5:
                    self.throttle = abs(1.0-self.steering)
                else:
                    self.throttle = abs(self.steering)

                # Clamp the throttle
                self.throttle = max(2*self.max_throttle*self.throttle,self.min_throttle)
            else:
                self.throttle = 0.0
        return self.steering, self.throttle

    def detect_line(self):
        ''' Update image from video source and fid centroid of mask.'''

        # Capture new image from source
        frame = self._capture_frame()
        mask = self._filter_frame()
        cnts = self._find_contours(mask)
        centroidX, centroidY = self._calculate_centroid(mask)
        print(f"{centroidX, centroidY}")
        steering, throttle = self._calculate_actuator_values(centroidX, centroidY)

        return frame, mask, steering, throttle, cnts, centroidX, centroidY

        # self.cX = int(self.frame.shape[0] / 2 - (self.cX - self.frame.shape[0] / 2))
    

    def draw_annotations(self, frame, cnts, centroidX, centroidY):
        frame = np.array(frame, dtype=np.uint8)
        # Draw centroids and contours on the frame
        max_area_contour = None
        max_area = -1

        for c in cnts:
            area = cv2.contourArea(c)
            # print(f'Contour area: {area}')
            if area > max_area:
                max_area = area
                max_area_contour = c

        if max_area_contour is not None:
            x, y, w, h = cv2.boundingRect(max_area_contour)
            print(f'Bounding box coordinates: x={x}, y={y}, width={w}, height={h}')
            cX = x + w // 2
            cY = y + h // 2
            center_point = (cX-(centroidX-cX), cY)
            frame = cv2.line(frame, (frame.shape[0] // 2, frame.shape[1]), center_point, (255,0,0), 9) 
            cv2.drawContours(frame, [max_area_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame,center_point , 7, (255, 255, 255), -1)
            cv2.putText(frame, "center", (cX-(centroidX-cX) - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if self.calibration_mode and (centroidX is not None) and (centroidY is not None):
            # print(f"{centroidX}, {centroidY}")
            cv2.circle(frame, (centroidX, centroidY), 5, (255, 255, 255), -1)
            cv2.putText(frame, "centroid", (centroidX - 25, centroidY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.drawContours(frame, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imshow(self.windowName, frame)
            
        return frame
    
    def _filter_frame(self):
        mask = self.hsv_filter(self.frame)

        if self.invert_mask:
            mask = cv2.bitwise_not(mask)
        
        # Crop the image
        # The crop values are in percentage of the original image, convert to pixels
        left_crop = int(self.left_crop * self.frame.shape[1])
        right_crop = int(self.right_crop * self.frame.shape[1])
        top_crop = int(self.top_crop * self.frame.shape[0])
        bottom_crop = int(self.bottom_crop * self.frame.shape[0])

        # zero out pixels not in the mask
        mask[0:top_crop,:] = 0
        mask[(mask.shape[0]-bottom_crop):mask.shape[0],:] = 0
        mask[:,0:left_crop] = 0
        mask[:,(mask.shape[1]-right_crop):mask.shape[1]] = 0
        cv2.imshow(self.maskWindowName, mask)
        return mask

    def hsv_filter(self, frame):
        ''' This method is encharged of searching for the line in the HSV color space'''
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.lower_hue, self.lower_sat, self.lower_val),
                                (self.upper_hue, self.upper_sat, self.upper_val))
    
        return mask

    def show_frame(self, frame, mask):
        cv2.imshow(self.windowName, frame)
        cv2.imshow(self.maskWindowName, mask)
        cv2.waitKey(1)

    def run(self):
        frame, mask, steering, throttle, cnts, centroidX, centroidY = self.detect_line()
        if frame is not None:  # Check if frame is not None
            print(f'Steering: {steering}, Throttle: {throttle}')
            frame_with_annotations = self.draw_annotations(frame, cnts, centroidX, centroidY)
            self.show_frame(frame_with_annotations, mask)
