import cv2
import numpy as np

'Our video to Select'
cap = cv2.VideoCapture('video.mp4') 

min_w_rec = 75  #min width of rectangle
min_h_rec = 75  #min height of rectangle
count_line_position = 500

'Initial Substractor'
algo = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=200, detectShadows=True) 

def center_handle(x,y,w,h):
    x1= int(w/2)
    y1= int(h/2)
    centerx= x+x1
    centery= y+y1
    return centerx,centery

detect = []
offset = 4 # line offset for pixel
counter = 0

while True:
    ret,frame1= cap.read()
    
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)

    '''Applying on each frame'''
    img_sub = algo.apply(blur)

    dilat = cv2.dilate(img_sub,np.ones((4,4)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    '''Selecting the line for counting -frame-x1 loc-x2 loc-color-thickness '''
    cv2.line(frame1, (60,count_line_position), (1200,count_line_position), (10,7,90), thickness=4)

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        val_counter = ((w>=min_w_rec) & (h>=min_h_rec))
        if not val_counter:
            continue

        cv2.rectangle(frame1, (x,y), (x+w,y+h), (87,90,7), thickness=2)

        center = center_handle(x,y,w,h)
        
        detect.append(center)
        cv2.circle(frame1,center,4,(0,255,0),-1)

    for(x,y) in detect:
        if y<(count_line_position+offset) and y>(count_line_position-offset):
            counter+=1
        cv2.line(frame1, (60,count_line_position), (1200,count_line_position), (53,54,208), thickness=3)
        detect.remove((x,y))
        print("The Cars That are Counted:"+str(counter))

    cv2.putText(frame1,("The Cars That are Counted:"+str(counter)),(450,70),cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)

    cv2.imshow('Detecter Filter',dilatada)

    cv2.imshow('Orjinal Video',frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()