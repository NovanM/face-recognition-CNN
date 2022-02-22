from datetime import datetime
import os
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


directory = 'Datasets10/train_new/'
dir = os.listdir(directory)
labels = []
for item in dir:
    if os.path.isdir(directory+item) == True:
        labels.append(item)
print(labels)
le = LabelEncoder()
labels = le.fit_transform(labels)

labels = le.classes_
print(labels)

HarCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

model = load_model('model_batch_face.h5')


def draw_ped(img, label, x0, y0, xt, yt, color=(0,0,0), text_color=(0,0,0)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
    cv2.rectangle(img,
                  (x0, y0 + baseline),  
                  (max(xt, x0 + w), yt), 
                  color, 
                  2)
    cv2.rectangle(img,
                  (x0, y0 - h),  
                  (x0 + w, y0 + baseline), 
                  color, 
                  -1)  
    cv2.putText(img, 
                label, 
                (x0, y0),                   
                cv2.FONT_HERSHEY_COMPLEX,     
                0.5,                          
                text_color,                
                1,
                cv2.LINE_AA) 
    return img

def attendence(n):
    with open("data_attendence.csv", "r+", newline="\n") as f:
        myDatalist = f.readlines()
        name_list = ''
        for line in myDatalist:
            name_list = line
        if n not in name_list:
            now = datetime.now()
            d1 = now.strftime("%d/%m/%Y")
            dtString = now.strftime("%H:%M:%S")
            data_name =  n,dtString,d1, "Present"
            if data_name not in f.readlines():
                f.writelines(f"\n{data_name}")
            else:
                pass


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    if ret : 
        # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = HarCascade.detectMultiScale(gray,1.1,4)

        for (x,y,w,h) in faces:           
            faces_img = gray[y:y+h,x:x+w]
            faces_img = cv2.resize(faces_img,(224,224))
            faces_img = faces_img.reshape(1,224,224,1)
            
            result = model.predict(faces_img)
            idx = result.argmax(axis=1)
            confidence = result.max(axis=1)*100
            label_txt = ''
            if confidence > 0:
                # label_txt = "%s"% (labels[idx])
                label_txt = "%s (%.2f %%)"% (labels[idx],confidence)
                attendence(label_txt)
                
                print(label_txt) 
            else:
                label_txt = "Uknown"

            frame = draw_ped(frame,label_txt,x,y,x+w,y+w,color=(0,255,255),text_color=(0,0,0))
        cv2.imshow("Detect Face",frame)
    else:
        break
    if cv2.waitKey(20) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()



