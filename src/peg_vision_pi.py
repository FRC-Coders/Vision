import sys
import cv2
import numpy as np
import JokerConfig as conf
#----------------------------------------------------------------------------------	
#capture configurations
cap = cv2.VideoCapture(0)
FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)
print "camera configured"

while 4320:
    print "working"
    #reading camera
    _,frame = cap.read()
    #hsv convert
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    print "frame readed as hsv"
    #bluring
    kernel = (19,19)
    blur = cv2.GaussianBlur(hsv, kernel,0)
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    print "blurred"
    #create mask
    low,high = conf.GREEN_RANGE
    mask = cv2.inRange(opening,low,high)
    #create res
    res = cv2.bitwise_and(frame,frame,mask=mask)
    print "mask created"

    
    
    #find contours
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        
    cnts = []
    #filter ration
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if (conf.MIN_RECT_RATIO<=float(h)/w<=conf.MAX_RECT_RATIO) and h*w>1000:
            cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cnts.append(c)
    #filter polygon
    epsil = 0.02
    goals = []
    for c in cnts:
        hull = cv2.convexHull(c)
        epsilon = epsil * cv2.arcLength(hull, True)
        goals.append(cv2.approxPolyDP(hull, epsilon, True))
    #calc centers by moments
    centers_x = [0]
    centers_y = [0]
    for g in goals:
        M = cv2.moments(g)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx,cy)
            cv2.circle(res, center, 5, (255, 0, 0), -1)
            centers_x.append(cx)
            centers_y.append(cy)
    #initialize angle
    angle = 0
    #drwing contours for goals centers
    cv2.drawContours(res, goals, -1, conf.DRAW_COLOR, 3)
    #intialize targets
    target_x = 0
    target_y = 0
    #calc centers of targets
    if len(centers_x) == 3 and len(centers_y) == 3:
        target_x = float(centers_x[1] + centers_x[2])/2
        target_y = float(centers_y[1] + centers_y[2])/2
        cv2.circle(res, (int(target_x),int(target_y)), 5, (0, 255, 0), -1)
    frame_cx,frame_cy = int(float(FRAME_WIDTH)/2),int(float(FRAME_HEIGHT)/2)
    cv2.circle(res, (frame_cx,frame_cy), 5, (255, 0, 0), -1)
    #calc error
    error = target_x - frame_cx
    angle = error * (float(conf.FOV_ANGLE) / FRAME_WIDTH)
    cv2.putText(res,str(angle),(40,240),4,1,(255,255,255))
    print "angle OK"
    #draw bounding rects and distance calc
    if len(goals) > 1 :
        x1,y1,w1,h1= cv2.boundingRect(goals[0])
        x2,y2,w2,h2= cv2.boundingRect(goals[1])
        cv2.putText(res,str((((2.0*714)/((w1+w2)/2.0))*conf.INCH2CM)),(40,140),4,1,(255,255,255))
        print "distance OK"
    #display
        
    #cv2.imshow("source",frame)
    #cv2.imshow("mask",mask)
    #cv2.imshow("cont",res)
    #cv2.imshow(window,img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
