import numpy as np
import cv2

def rectangle(img): #함수로 만든다
#파란색
	x1 = 280; y1 = 80; #자르고 싶은 지점의 x좌표와 y좌표 지정
	w1 = 65; h1 = 70; #x로부터 width, y로부터 height를 지정

#초록색
	x2 = 10; y2 = 156; #자르고 싶은 지점의 x좌표와 y좌표 지정
	w2 = 124; h2 = 119; #x로부터 width, y로부터 height를 지정

#빨간색
	x3 = 170; y3 = 700; #자르고 싶은 지점의 x좌표와 y좌표 지정
	w3 = 72; h3 = 50; #x로부터 width, y로부터 height를 지정

#노란색
	x4 = 478; y4 = 416; #자르고 싶은 지점의 x좌표와 y좌표 지정
	w4 = 50; h4 = 100; #x로부터 width, y로부터 height를 지정
	img_trim =  cv2.rectangle(img, (x1,y1), (x1+w1, y1+h1), (255,0,0),2)
	img_trim =  cv2.rectangle(img, (x2,y2), (x2+w2, y2+h2), (0,255,0),2)
	img_trim =  cv2.rectangle(img, (x3,y3), (x3+w3, y3+h3), (0,0,255),2)
	img_trim =  cv2.rectangle(img, (x4,y4), (x4+w4, y4+h4), (0,255,255),2)



	return img_trim #필요에 따라 결과물을 리턴


image =cv2.imread('resize/a/1.jpg')
rectangle_image = rectangle(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('rectangle/1.jpg',rectangle_image)
