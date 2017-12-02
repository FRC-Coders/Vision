import cv2
import numpy as np
import Tkinter as tk
import TheJokerLib_Vision as jk
from lib import JokerConfig as conf
cap = cv2.VideoCapture(0)

while 4320:
    #reading camera
    _,frame = cap.read()
    
    #hsv convert
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #bluring
    blur = cv2.blur(hsv,conf.BLUR_KERNEL)
    #create mask
    mask = cv2.inRange(blur, conf.MIN_GREEN,MAX_GREEN)

    mask = jk.dil_ero(mask,(300,300),3)
    #create res
    res = cv2.bitwise_and(frame,frame,mask=mask)

    
    
    #find contours
    _,contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        
    cnts = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (conf.MIN_RECT_RATIO<=float(h)/w<=conf.MAX_RECT_RATIO) and h*w>1000:
            cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cnts.append(c)
    cv2.drawContours(res, cnts, -1, conf.DRAW_COLOR, 3)
    #draw a center plus
    if len(cnts) > 1 :
        x1,y1,w1,h1= cv2.boundingRect(cnts[0])
        x2,y2,w2,h2= cv2.boundingRect(cnts[1])
     #   pt1 = ((x1+w1/2+x2+w2/2)/2)
      #  pt2 = ((y1+h1/2+y2+h2/2)/2)
       # cv2.line(res,(pt1 + 7,pt2),(pt1 - 7, pt2),(0,0,255),2)
        #cv2.line(res,(pt1, pt2+7),(pt1,pt2-7),(0,0,255),2)
        cv2.putText(res,str((((2.0*714)/((w1+w2)/2.0))*conf.INCH2CM)),(x1,y1-20),1,5,(255,255,255))
    #display
    cv2.imshow("source",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("cont",res)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
