Chapter3. 신경망

1.	퍼셉트론에서 신경망으로
1-1.	신경망의 예/ 1-2. 퍼셉트론 복습/ 1-3 활성화 함수의 등장

신경망 퍼셉트론의 단점인 가중치 설정하는 작업을 자동으로 학습하여 수행한다

<img width="235" alt="image" src="https://user-images.githubusercontent.com/91449518/179902036-25d9a679-223f-46b5-821d-02706c9d571f.png">

신경망은 입력층, 은닉층, 출력층으로 구분된다

입력층: 입력받는 층

은닉층: 입력층과 출력층 사이의 층으로 사람눈에 보이지 않는 층

출력층: 출력하는 층

<img width="148" alt="image" src="https://user-images.githubusercontent.com/91449518/179902053-30441de6-f42c-4293-b1d7-01f2d39ec875.png">

활성화 함수: 입력 신호의 총합을 출력 신호로 변환하는 함수 ex) 위의 h()

2.	활성화 함수
2-1 시그모이드 함수/ 2-4 시그모이드 함수 구현하기

 
<img width="274" alt="image" src="https://user-images.githubusercontent.com/91449518/179902386-f64a8af1-ab27-45ad-9fc9-aab4b651122c.png">

 

2-2 계단 함수 수현하기/ 2-3 계단 함수의 그래프

 <img width="230" alt="image" src="https://user-images.githubusercontent.com/91449518/179902397-ed9565c9-0e7f-4a86-8581-42425eb71d2c.png">


2-5 시그모이드 함수와 계단 함수 비교

차이점: 시그모이드 함수는 부드러운 곡선이며 0과 1사이의 함수 반환
		계단 함수는 불연속적이며 0 혹은 1만 반환

공통점: 입력값이 작으면 0에 가깝게 크면 1에 가깝게 출력
		출력값이 0과 1 사이

2-6 비선형 함수

비선형 함수를 활성화 함수로 쓰는 이유는 선형 함수의 경우 층을 여러개로 해도 은닉층이 없는 네트워크로 표현이 가능하기에 층을 쌓는 혜택을 얻기 위해서이다

2-7 ReLU 함수

 <img width="167" alt="image" src="https://user-images.githubusercontent.com/91449518/179902431-8f221dfe-814f-4b03-8fc0-ce9a72d2d770.png">

 <img width="136" alt="image" src="https://user-images.githubusercontent.com/91449518/179902454-d5cae39a-a4c7-42f7-9d2f-4c200cc1f912.png">

 
3.	다차원 배열의 계산
3-1 다차원 배열/ 3-2 행렬의 곱/ 3-3 신경망에서의 행렬 곱


<img width="273" alt="image" src="https://user-images.githubusercontent.com/91449518/179902553-11f35115-59c7-4745-81b3-6d7a47de14a1.png">

 
입력을 받아 출력을 계산하는 신경망에서의 과정을 행렬의 곱으로 표현

4. 3층 신경망 구현하기

넘파이 사용시 적은 코드만으로 신경망의 순방향 처리를 완성할 수 있음 

<img width="211" alt="image" src="https://user-images.githubusercontent.com/91449518/179902594-89529aa8-bc69-47c5-abb5-e159d2a32639.png">


4-1 표기법 설명

<img width="234" alt="image" src="https://user-images.githubusercontent.com/91449518/179902633-1c4672b5-9b0a-453e-8d8e-0547d052571c.png">

4-2 각 층의 신호 전달 구현하기/ 4-3 구현 정리

<img width="199" alt="image" src="https://user-images.githubusercontent.com/91449518/179902641-e204e3e6-ad13-47f3-8702-a5973ec195ef.png">

위 그림의 두꺼운 선을 따라 1층의 1번째 뉴런의 a값을 아래의 행렬 이용해 식을 나타내면 A(1) = XW(1) + B(1) 
 
<img width="218" alt="image" src="https://user-images.githubusercontent.com/91449518/179902662-2d5791b6-a16d-4474-9471-20dfebbd35a1.png">
 

5 출력층 설계하기

활성화 함수: 회기 -> 항등함수, 분류 -> 소프트맥스 함수

5.1 항등 함수와 소프트맥스 함수 구현하기

항등 함수: 입력신호 그대로 출력

소프트맥스 함수

<img width="187" alt="image" src="https://user-images.githubusercontent.com/91449518/179902689-4f6e1fb4-5ff4-4788-aeee-7db335fe74e8.png">

 
5.2 소프트맥스 함수 구현 시 주의점
 
<img width="264" alt="image" src="https://user-images.githubusercontent.com/91449518/179902705-4ae0b5ee-3bb1-41b2-9e67-1aef11ec1068.png">


지수함수 꼴이라 수가 매우 커져서 오류를 발생시킬 수 있으므로 각 지수함수의 밑 부분에 같은 수를 더하면서 오류가 안 나게 해야 한다.

5.3 소프트맥스 함수의 특징

소프트맥스 함수의 출력은 0에서 1사이의 실수이다
출력의 총합은 1이된다
확룰로 해석 가능하다.
지수 함수가 단조 증가 함수이므로 각 원소의 대소 관계는 변하지 않는다

5.4 출력층의 뉴런 수 정하기

분류에서는 분류하고 싶은 클래스 수로 정한다

6 손글씨 숫자 인식

입력 데이터를 분류하는 추론 과정을 신견망의 순전파라고 한다

6.1 MNIST 데이터/ 6.2 신경망의 추론 처리/ 6.3 배치 처리

MNIST 데이터의 입력이 28*28의 크기이므로 입력층 뉴런은 784개이고 출력층 뉴런은 10개의 숫자로 분류하므로 10개이다

신경망 추론을 통해 정확도 평가

정규화: 데이터를 특정 범위로 변환하는 처리

전처리: 입력 데이터에 특정 변환을 가하는 것

배치(batch): 하나로 묶은 입력 데이터

배치 처리의 장점 
1. 이미지 1장당 처리 시간을 대폭 줄여준다
2. 버스에 주는 부하를 줄인다

7 정리

신경망에서는 활서화 함수로 시그모이드함수와 ReLU함수 같은 매끄럽게 변화하는 함수 이용

넘파이의 다차원 배열을 잘 사용해 신경망을 효율적으로 구현

기계학습 문제는 크게 회귀와 분류로 나뉨

출력층의 활성화 함수로는 회귀 -> 항등함수 분류 -> 소프트맥스 함수 이용

분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수로 설정

입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 배치 단워로 진행시 훨씬 빠르게 결과 얻을 수 있다



