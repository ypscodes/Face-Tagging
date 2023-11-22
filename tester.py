import cv2
import os
import numpy as np
import faceRecognition as fr


#This module takes images stored in diskand performs face recognition
# test_img=cv2.imread('TestImages/Y1.jpg')#test_img path for Yash
test_img=cv2.imread('TestImages/R5.jpeg')#test_img path for Riya
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)


# Comment belows lines when running this program second time.Since it saves training.yml file in directory
# faces,faceID=fr.labels_for_training_data('trainingImages')
# face_recognizer=fr.train_classifier(faces,faceID)
# face_recognizer.write('trainingData.yml')


#Uncomment below line for subsequent runs
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')#use this to load traininADg data for subsequent runs

name={0:"Priyanka",1:"Kangana",2:"Yash",3:"Riya"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("Confidence:",100 - confidence)
    print("Label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("Face Tagging Project",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows