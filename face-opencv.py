import cv2
import time
import csv
from numpy import ERR_IGNORE
import pandas as pd
from datetime import datetime, date

from pandas.core.frame import DataFrame

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
df = pd.DataFrame();
def create_dataset(img,id,img_id):
        cv2.imwrite("data/pic."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,clf):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
        coords=[]
        global df
        for (x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w,y+h),color,2)
            id,con= clf.predict(gray[y:y+h,x:x+w])#ทำนายเปอร์เซ็นความแม่นยำ
            
            
            if con <= 100:
                    cv2.putText(img,"Hello", (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
            else:
                    cv2.putText(img,"Unknow", (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

            if (con < 100):
                    con = "  {0}".format(round(100 - con))
                    x = int(con)
                    if x >= 60:
                        people = "Audy"
                        d = date(2021, 3, 18)
                        now = datetime.today()
                        today = date.today()
                        time_day = d
                        dt_string = now.strftime("%H:%M:%S")
                        df = df.append({'A':people,'B':time_day,'C':dt_string,'D':con}, ignore_index=True)
                        print(df)
            else:
                    con = "  {0}%".format(round(100 - con))
            cv2.putText(img,str(con), (x+25,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)


            coords=[x,y,w,h]
            
        return img,coords


def detect(img,faceCascade,img_id,clf):
        img,coords=draw_boundary(img,faceCascade,1.1,10,(0,255,0),clf)
        if len(coords) == 4:
            id=1
            result = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
            #create_dataset(result,id,img_id)

        return img
        
img_id = 0
cap = cv2.VideoCapture(0) #เปลี่ยนเป็นชื่อวีดีโอ หรือใน 0 แทนเพื่อเปิดกล้อง

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

start_time = time.time()
while (True):
        ret,frame = cap.read()
        frame=detect(frame,faceCascade,img_id,clf)
        cv2.imshow('fram',frame)
        img_id += 1
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            df.to_csv('my_csv.csv', mode='a+',index=False,header=False )
            break

cap.release()
cv2.destroyAllWindows()
