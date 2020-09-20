import cv2 
import dlib
import math
from playsound import playsound


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Machine_learning/cv/shape_predictor_68_face_landmarks.dat")

def getmid(p1,p2):
    return int((p1.x + p2.x)/2),int((p1.y + p2.y)/2)

def getratio(points,f_landmark):
        horizontal_left = (landmarks.part(points[0]).x,landmarks.part(points[0]).y)
        horizontal_right = (landmarks.part(points[3]).x,landmarks.part(points[3]).y)
        cv2.line(frame,horizontal_left,horizontal_right,(0,0,255),1)

        vertical_top = getmid(landmarks.part(points[1]),landmarks.part(points[2]))
        vertical_bottom = getmid(landmarks.part(points[4]),landmarks.part(points[5]))
        cv2.line(frame,vertical_top,vertical_bottom,(0,0,255),1)

        hori = math.hypot((horizontal_right[0]-horizontal_left[0]),(horizontal_right[1]-horizontal_left[1]))  
        ver = math.hypot((vertical_top[0]-vertical_bottom[0]),(vertical_top[1]-vertical_bottom[1])) 

        return hori/ver 

def isblinking(p1,p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        print("blink")
while True:
    _,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray,face)
        left_ratio = getratio((36,37,38,39,40,41),landmarks)
        right_ratio = getratio((42,43,44,45,46,47),landmarks)

        if (left_ratio + right_ratio)/2 > 6.5:
            playsound('D:/Machine_learning/cv/tink.wav')
            cv2.putText(frame,"Blink",(50,150),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
        #isblinking(vertical_top,vertical_bottom)
    cv2.imshow("Main",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()