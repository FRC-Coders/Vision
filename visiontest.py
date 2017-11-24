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
    lower_green = np.array([0,0,100])
    upper_green = np.array([255,150,255])
    
    #create mask
    mask = cv2.inRange(blur, lower_green,upper_green)
    #create res
    res = cv2.bitwise_and(frame,frame,mask=mask)
    #create threshold
    ret,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    
    #find contours
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res, contours, -1, (45,255,30), 3)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #draw a center plus
    if len(contours) == 2:
        x1,y1,w1,h1= cv2.boundingRect(contours[0])
        x2,y2,w2,h2= cv2.boundingRect(contours[1])
        pt1 = ((x1+w1/2+x2+w2/2)/2)
        pt2 = ((y1+h1/2+y2+h2/2)/2)
        cv2.line(res,(pt1 + 7,pt2),(pt1 - 7, pt2),(0,0,255),2)
        cv2.line(res,(pt1, pt2+7),(pt1,pt2-7),(0,0,255),2)
    #display
    cv2.imshow("source",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("cont",res)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
