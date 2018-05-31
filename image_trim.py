import numpy as np
import cv2


def im_trim (img): #함수로 만든다
#e2z 3
    # x = 280; y = 80; #자르고 싶은 지점의 x좌표와 y좌표 지정
    # w = 65; h = 80; #x로부터 width, y로부터 height를 지정

#1
    x = 10; y = 156; #자르고 싶은 지점의 x좌표와 y좌표 지정
    w = 124; h = 119; #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    return img_trim #필요에 따라 결과물을 리턴


#4
#     x = 170; y = 700; #자르고 싶은 지점의 x좌표와 y좌표 지정
#     w = 72; h = 50; #x로부터 width, y로부터 height를 지정

# #2
#     x = 478; y = 416; #자르고 싶은 지점의 x좌표와 y좌표 지정
#     w = 50; h = 100; #x로부터 width, y로부터 height를 지정


#     img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
#     return img_trim #필요에 따라 결과물을 리턴

image =cv2.imread('resize/a/1.jpg')
trim_image = im_trim(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('trim/1_osap/1.jpg',trim_image)

image =cv2.imread('resize/a/2.jpg')
trim_image = im_trim(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('trim/1_osap/2.jpg',trim_image)

image =cv2.imread('resize/a/3.jpg')
trim_image = im_trim(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('trim/1_osap/3.jpg',trim_image)

image =cv2.imread('resize/a/4.jpg')
trim_image = im_trim(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('trim/1_osap/4.jpg',trim_image)

image =cv2.imread('resize/a/5.jpg')
trim_image = im_trim(image) #trim_image 변수에 결과물을 넣는다
cv2.imwrite('trim/1_osap/5.jpg',trim_image)