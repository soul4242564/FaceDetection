import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("dataSet/recognizer/trainner.yml")
labels = {"Name": 1}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret,image=cap.read();
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w] #(ycord1,ycord_end)
        roi_color = image[y:y+h,x:x+w]
        #reconizer
        id_,conf = recognizer.predict(roi_gray)
        if conf>=45 and conf <=85:
            print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(image, name, (x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item = "my-img.jpg"
        cv2.imwrite(img_item, color)

        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_Y = y+h
        cv2.rectangle(image,(x,y),(end_cord_x,end_cord_Y),color,stroke)

        
    cv2.imshow("Face",image);
    if  0xff == ord('q'):
            break;

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
