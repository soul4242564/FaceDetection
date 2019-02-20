import cv2
import os
import numpy as np
from PIL import Image
import pickle


faceDetect=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml');
cap=cv2.VideoCapture(0);

id=input('Enter your Name : ')
idNum=0;
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
while(True):
    createFolder('./dataSET/user/'+str(id)+'/')
    ret,frame=cap.read();
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        if conf>=45 and conf <=70:
            idNum=idNum+1;
            cv2.imwrite("dataSet/User/"+str(id)+"/"+str(id)+"."+str(idNum)+".jpg",gray[y:y+h,x:x+w]);
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.waitKey(100);
        cv2.imshow("Face",frame);
        cv2.waitKey(1);
        if(idNum>19):
            break

cap.release()
cv2.destroyAllWindows()



#Recognizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"dataSET/User")

faceDetect=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml');
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []
for root,dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace("","").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id += 1
            id_ =label_ids[label]
            print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image,"uint8")
            print(image_array)
            faces = faceDetect.detectMultiScale(image_array,1.3,5);

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_labels))
recognizer.save("dataSet/recognizer/trainner.yml")





