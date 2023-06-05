import cv2
from pose_module import poseDetector
import time

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