# MakIT/마킷


![1](https://user-images.githubusercontent.com/29969821/42430604-30a1d97a-837b-11e8-947f-c9093d7d86af.jpg)

성공하다 라는 의미의 Make It에서 Error를 뜻하는 e를 제거한 의미로 머신 러닝을 통해 SMD 불량을 제거.

  - 개발 배경
  - 개발 Flow
  - 사용 기술
  - 프로그램 출력 결과


# 개발 배경

  - SMD 공정이란 기판 표면에 부품을 실장하는 방식으로 전자기기의 소형화 및 제조공정의 발전으로 제조과정에서 널리 사용. 
 - 하지만 공정조건 변화에 따라 오삽, 미삽, 회전이라는 불량이 발생.
 - 이러한 SMD 공정에서 불량을 검사하기 위해 현재 부품별로 육안으로 확인해야 되고, 각 부품의 패키지 종류에 따라 수동으로 등록.
 - 그래서 시간과 노력이 많이 소요되는 문제점 때문에 부품 패키지 분류 과정에 대한 자동화가 필요했습니다.

![2](https://user-images.githubusercontent.com/29969821/42430612-38e8c076-837b-11e8-9e15-d074a210503a.jpg)

따라서 :
  - 이러한 문제점을 개선하기 위해 자동화 시스템인 마킷을 개발
  - 마킷의 빠르고 정확한 검사를 통해 자동으로 불량 원인을 파악하고 이에 따라 최적의 가치를 창출하고자 하는 목적.


# 개발 Flow

![3](https://user-images.githubusercontent.com/29969821/42430102-140baeb0-8378-11e8-940c-c6ac533b8939.jpg)

1. image_trim.py : 이미지 데이터를 Reszie

2. image_scan.py : 이미지 데이터를 전처리화 해서 노이즈 제거 & 에지 추출
 
3. object_extraction.py : 부품 별로 객체 영역 추출

4. inception_v3.py : Inception_V3을 통한 트레이닝

5. test_driver.py : 검사하고자 하는 부품 이지미 검사 와 결과 


# 사용 기술 - Inception V3

![4](https://user-images.githubusercontent.com/29969821/42430184-a0b94372-8378-11e8-99d9-c05499cdf66c.jpg)

매번 새로운 Task에 대해 바닥부터 적합한 Neural Networks 구조를 발견하고, 이를 새로 학습시키는 것은 매우 많은 시간과 노력이 들어가는 일.
따라서, 특정 Task에 이미 잘 작동하는 것이 검증된 모델-예를 이용한 Transfer Learning를 이용.
이미지 인식(추론)에 대해 구글이 만든 Inception 모델-의 모델 구조와 파라미터들을 이용
이를 기반으로 새로운 데이터셋에 retraining을 진행해주어 효율성을 높였다.


# 프로그램 출력 결과

![5](https://user-images.githubusercontent.com/29969821/42430185-a2dcae6e-8378-11e8-874b-e024392c3d60.jpg)

![6](https://user-images.githubusercontent.com/29969821/42430188-a46e021e-8378-11e8-8997-a550736a59f3.png)

실제 부품의 불량검사에 대한 정확한 결과를 출력
