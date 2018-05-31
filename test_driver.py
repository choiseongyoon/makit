# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""
# python retrain_run_inference.py

import numpy as np
import tensorflow as tf
import cv2

# 파란색
x1 = 280;
y1 = 80;  # 자르고 싶은 지점의 x좌표와 y좌표 지정
w1 = 65;
h1 = 70;  # x로부터 width, y로부터 height를 지정

# 초록색
x2 = 10;
y2 = 156;  # 자르고 싶은 지점의 x좌표와 y좌표 지정
w2 = 124;
h2 = 119;  # x로부터 width, y로부터 height를 지정

# 빨간색
x3 = 170;
y3 = 700;  # 자르고 싶은 지점의 x좌표와 y좌표 지정
w3 = 72;
h3 = 50;  # x로부터 width, y로부터 height를 지정

# 노란색
x4 = 478;
y4 = 416;  # 자르고 싶은 지점의 x좌표와 y좌표 지정
w4 = 50;
h4 = 100;  # x로부터 width, y로부터 height를 지정


def im_trim1(img):  # 함수로 만든다
    img_trim1 = img[y1:y1 + h1, x1:x1 + w1]  # trim한 결과를 img_trim에 담는다
    return img_trim1  # 필요에 따라 결과물을 리턴


def im_trim2(img):  # 함수로 만든다
    img_trim2 = img[y2:y2 + h2, x2:x2 + w2]  # trim한 결과를 img_trim에 담는다
    return img_trim2  # 필요에 따라 결과물을 리턴


def im_trim3(img):  # 함수로 만든다
    img_trim3 = img[y3:y3 + h3, x3:x3 + w3]  # trim한 결과를 img_trim에 담는다
    return img_trim3  # 필요에 따라 결과물을 리턴


def im_trim4(img):  # 함수로 만든다
    img_trim4 = img[y4:y4 + h4, x4:x4 + w4]  # trim한 결과를 img_trim에 담는다
    return img_trim4  # 필요에 따라 결과물을 리턴


image = cv2.imread('C:/Users/ChoiSeongYoon/python/resize/e2z_shift/2.jpg')
trim_image1 = im_trim1(image)  # trim_image 변수에 결과물을 넣는다
cv2.imwrite('C:/Users/ChoiSeongYoon/python/test1.jpg', trim_image1)
# trim_image2 = im_trim2(image) #trim_image 변수에 결과물을 넣는다
# cv2.imwrite('C:/Users/ChoiSeongYoon/python/test2.jpg',trim_image2)
# trim_image3 = im_trim3(image) #trim_image 변수에 결과물을 넣는다
# cv2.imwrite('C:/Users/ChoiSeongYoon/python/test3.jpg',trim_image3)
# trim_image4 = im_trim4(image) #trim_image 변수에 결과물을 넣는다
# cv2.imwrite('C:/Users/ChoiSeongYoon/python/test4.jpg',trim_image4)

imagePath = 'C:/Users/ChoiSeongYoon/python/test1.jpg'
# imagePath = 'C:/Users/ChoiSeongYoon/python/test2.jpg'
# imagePath = 'C:/Users/ChoiSeongYoon/python/test3.jpg'
# imagePath = 'C:/Users/ChoiSeongYoon/python/test4.jpg'

# imagePath = 'C:/Users/ChoiSeongYoon/python/f_e2z2.jpg'
modelFullPath = 'C:/Users/ChoiSeongYoon/python/tmp/output_graph.pb'
labelsFullPath = 'C:/Users/ChoiSeongYoon/python/tmp/output_labels.txt'


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(imagePath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-1:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.

        # top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
        answer = labels[top_k[0]]
        return answer


def rectangle(img):  # 함수로 만든다
    img_rectangle = cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
    img_rectangle = cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
    img_rectangle = cv2.rectangle(img, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2)
    img_rectangle = cv2.rectangle(img, (x4, y4), (x4 + w4, y4 + h4), (0, 255, 255), 2)
    return img_rectangle  # 필요에 따라 결과물을 리턴


image = cv2.imread('C:/Users/ChoiSeongYoon/python/resize/e2z_shift/2.jpg')
trim_image1 = im_trim1(image)  # trim_image 변수에 결과물을 넣는다
cv2.imwrite('C:/Users/ChoiSeongYoon/python/test1.jpg', trim_image1)
trim_image2 = im_trim2(image)  # trim_image 변수에 결과물을 넣는다
cv2.imwrite('C:/Users/ChoiSeongYoon/python/test2.jpg', trim_image2)
trim_image3 = im_trim3(image)  # trim_image 변수에 결과물을 넣는다
cv2.imwrite('C:/Users/ChoiSeongYoon/python/test3.jpg', trim_image3)
trim_image4 = im_trim4(image)  # trim_image 변수에 결과물을 넣는다
cv2.imwrite('C:/Users/ChoiSeongYoon/python/test4.jpg', trim_image4)

imagePath1 = 'C:/Users/ChoiSeongYoon/python/test1.jpg'
imagePath2 = 'C:/Users/ChoiSeongYoon/python/test2.jpg'
imagePath3 = 'C:/Users/ChoiSeongYoon/python/test3.jpg'
imagePath4 = 'C:/Users/ChoiSeongYoon/python/test4.jpg'

if __name__ == '__main__':
    run_inference_on_image(imagePath1)

    run_inference_on_image(imagePath2)

    run_inference_on_image(imagePath3)

    run_inference_on_image(imagePath4)

    rectangle_image = rectangle(image)  # trim_image 변수에 결과물을 넣는다

    cv2.putText(rectangle_image, "Shift", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
    cv2.putText(rectangle_image, "Adjust", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
    cv2.putText(rectangle_image, "Adjust", (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv2.putText(rectangle_image, "Adjust", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))

    cv2.imwrite('rectangle/1.jpg', rectangle_image)