import cv2
import numpy as np
import Tkinter as tk
cap = cv2.VideoCapture(0)

while True:
    #reading camera
    _,frame = cap.read()
    
    #hsv convert
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #bluring
    blur = cv2.blur(hsv,(5,5))
    #define color range
    lower_green = np.array([40,90,90])
    upper_green = np.array([70,255,255])
    
    #create mask
    mask = cv2.inRange(blur, lower_green,upper_green)
    flip_mask = cv2.bitwise_not(mask)#flip mask colors
    #create res
    res = cv2.bitwise_and(frame,frame,mask=mask)
    #create threshold
    ret,thresh = cv2.threshold(flip_mask,127,255,0)
    
    #find an draw contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, (0,255,0), 3)
    #finding max contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    #drawing rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
    #uncomment for blobs
    #detector = cv2.SimpleBlobDetector()
    #keypoints = detector.detect(flip_mask)
    #im_with_keypoints = cv2.drawKeypoints(res, keypoints, np.array([]),(0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #display
    cv2.imshow("source",frame)
    #cv2.imshow("blob",im_with_keypoints)
    cv2.imshow("mask",flip_mask)
    cv2.imshow("cont",res)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
