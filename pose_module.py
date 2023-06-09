import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, static_image_mode = False, model_complexity = 1,
                 smooth_landmarks = True,
                 enable_segmentation = False, smooth_segmentation = True,
                 min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
                
                self.statatic_image_mode = static_image_mode
                self.model_complexity = model_complexity
                self.smooth_landmarks = smooth_landmarks
                self.enable_segmentation = enable_segmentation
                self.smooth_segmentation = smooth_segmentation
                self.min_detection_confidence = min_detection_confidence
                self.min_tracking_confidence = min_tracking_confidence

                self.mpDraw = mp.solutions.drawing_utils
                self.mpPose = mp.solutions.pose
                self.pose = self.mpPose.Pose(self.statatic_image_mode,
                                        self.model_complexity,
                                        self.smooth_landmarks,
                                        self.enable_segmentation,
                                        self.smooth_segmentation,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
                
    def findPose(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        

        return img
    
    def getPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture('videos/vid3.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        detector.findPose(img)
        lmList = detector.getPosition(img, draw = False)
        print(lmList[14])
        cv2.circle(img, (lmList[25][1], lmList[25][2]), 10, (255, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        
        cv2.putText(img, str(fps), (70,50), cv2.FONT_HERSHEY_PLAIN,3,
                    (255,0,0), 3)
        
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', img)
        key = cv2.waitKey(20)
        if key > 0:
            break

if __name__== '__main__':
    main()

