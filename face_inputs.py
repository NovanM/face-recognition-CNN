import cv2
import os

cap = cv2.VideoCapture(0)

# Input Name Folder for class dataset
my_name = "Name Dataset/Classs"

dataset_folder = "Datasets10/face_inputs/"
os.mkdir(dataset_folder + my_name)
num_sample = 50
i = 0
while cap.isOpened():
    ret, frame = cap.read()   
    if ret :
        cv2.imshow("Capture Photo", frame)
        cv2.imwrite("Datasets10/face_inputs/%s/%s_%04d.jpg" %  (my_name, my_name, i), cv2.resize(frame, (250,250)))
        
        if cv2.waitKey(50) == ord('q') or i == num_sample:
            break
        i += 1    
cap.release()
cv2.destroyAllWindows()
