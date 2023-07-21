import numpy as np
import cv2
import face_recognition
#使用face库加载图像获取RGB，注意获取到的为BGR，需要转化为RGB
imgElon=face_recognition.load_image_file('ImageBasic/Elon Musk.png')
#BGR转化为RGB
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

#使用face库加载图像获取RGB，注意获取到的为BGR，需要转化为RGB
imgTest=face_recognition.load_image_file('ImageBasic/Elon Test.png')
#BGR转化为RGB
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#检测面部，提取特征，特征编码
#获取面部位置，返回值为人脸框的坐标
faceLoc=face_recognition.face_locations(imgElon)[0]
print(faceLoc)
#面部特征编码
encodeElon=face_recognition.face_encodings(imgElon)[0]
#左上角右下角坐标cv2.rectangle(imgElon, (faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,255),2)

#获取面部位置，返回值为人脸框的坐标
faceLocTest=face_recognition.face_locations(imgTest)[0]
print(faceLocTest)
#面部特征编码
encodeTest=face_recognition.face_encodings(imgTest)[0]
#左上角右下角坐标
cv2.rectangle(imgTest, (faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#比较两张图片的面部特征，列表处可以填写多张图片表示目前已知的人脸库，test表示要测试的人脸图片，返回值是一个列表，若返回True则表示特征相近为同一个人
result=face_recognition.compare_faces([encodeElon],encodeTest)
print(result)
#定量的给出两个人脸的差距，差距越小越好，返回值为一个列表
faceDis=face_recognition.face_distance([encodeElon],encodeTest)
print(faceDis)

cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
cv2.imshow('Elon Musk',imgElon)
cv2.imshow('Elon Test',imgTest)
cv2.waitKey(0)