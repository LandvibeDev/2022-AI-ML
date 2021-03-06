# 6 학습 관련 기술들
이번 장에서 다룰 주제
- 가중치 매개변수의 최적값을 탐색하는 최적화 방법
- 가중치 매개변수 초깃값, 하이퍼파라미터 설정 방법
- 오버피팅의 대응책인 가중치 감소와 드롭아웃 등의 정규화 방법
- 배치 정규화
## 6.1 매개변수 갱신
신경망 학습의 목적은 손실함수의 값을 가능한 낮추는 매개변수의 최적값을 찾는 것이었음\
앞에서는 최적의 매개변수 값을 찾는 단서로 매개변수의 기울기를 이용해,\
기울어진 방향으로 매개변수 값을 갱신하는 방법인 <stong>확률적 경사 하강법(SGD)</strong>을 이용했음\
### 6.1.1 모험가 이야기
```
가장 깊고 낮은 골짜기에서 가장 깊고 낮은 골짜기를 찾아가려함.
엄격한 '제약' 2개가 존재.
하나는 지도를 보지 않을 것, 또 하나는 눈가리개를 쓰는 것.
어떻게 '깊은 곳'을 찾을 수 있을까?
```
이 상황에서 '기울기'라는 단서로 가장 크게 기울어진 방향으로 가는것이 SGD의 전략임

### 6.1.2 확률적 경사 하강법(SGD)
SGD의 수식은 다음과 같음

$$W \leftarrow W - \eta \frac{dL}{dW}$$

$W$는 갱신할 가중치 매개변수, $\frac{dL}{dW}$는 $W$에 대한 손실함수의 기울기, $\eta$는 학습률을 뜻함\
식을 그대로 해석하면 기울어진 방향으로 일정거리 $\eta$만큼 이동하는 단순한 방법임\
\
이를 파이썬 클래스로 구현하면
```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```
lr은 학습률, params는 가중치 매개변수, grads는 기울기를 저장하고 있음\
위 SGD 클래스를 이용하여 신경망 매개변수의 진행을 다음과 같이 수행할 수 있음
```python
network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000):
    ...
    x_batch, t_batch = get_mini_batch(...) #미니배치
    grads = network.gradient(x_batch, t_batch)
    params = network.params
    optimizer.update(params, grads)
    ...
```
만약에 최적화 기법이 바뀌더라도 위 예시에서 optimizer 부분만 바꿔주면 됨
### 6.1.3 SGD의 단점
$f(x,y)=\frac{1}{20}x^2 +y^2$[식 6.2]
함수의 최솟값을 구하는 문제를 생각해보면
![img6-1](https://user-images.githubusercontent.com/58386334/180708981-f2b67eb6-d17a-42d5-9b55-8a01f0982974.jpeg)
위와 같이 같이 그래프와 등고선이 그려지게 됨\
SGD는 기울기의 방향으로 갱신을 하는데, $f(x,y)$의 기울기가 아래와 같다
![Img6-2](https://user-images.githubusercontent.com/58386334/180709078-fec828b2-90e9-4d51-aea4-cc4553e52d2d.jpeg)
따라서 각 지점에서 y축 방향의 기울기는 크지만 x축 방향의 기울기는 작음\
그로부터 각 지점에서 기울기는 대부분 (0,0)을 가리키지 않음을 알 수 있음\
따라서 임의의 탐색 시작 지점에서 함수에 SGD를 적용해보면 아래와 같다
<img width="652" alt="img6-3" src="https://user-images.githubusercontent.com/58386334/180709092-145f1575-14ad-43b9-93bc-1a42df34c512.png">

위의 그림을 통해, 해당 예시에서 SGD는 비효율적인 움직임을 가짐을 알 수 있음\
\
SGD가 지그재그로 탐색하는 근본 원인은 기울어진 방향이 본래의 최솟값과 다른 방향을 가리킨다는 점으로 생각할 수 있음\
\
이러한 SGD의 단점을 개선해주는 모멘텀, AdaGrad, Adam 세가지 방법이 존재함
### 6.1.4 모멘텀
모멘텀은 물리에서의 <strong>'운동량'</strong>을 뜻하는 단어임\
모멘텀 기법의 수식은 다음과 같이 쓸 수 있음\

$$
\begin{aligned}
v\leftarrow \alpha v-\eta \frac{dL}{dW} [식 6.3]\\
W\leftarrow W+v[식 6.4]
\end{aligned}
$$

여기서 추가된 $v$는 물리에서의 속도에 해당함\
[식 6.3]은 기울기 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타냄\
[식 6.3]의 $\alpha v$ 항은 물체가 아무런 힘을 받디 않을 때 서서히 하강 시키는 역할을 함\
모멘트를 구현하면 아래와 같음
```python
class Momentum:
    def __init__(self, lr=0.01, mementum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.b[key] - self.lr * grads[key]
            params[key] += self.v[key]
```
모멘텀을 이용하여 최적화 문제를 풀어보면 다음과 같은 결과를 얻을 수 있다
<img width="921" alt="img6-5" src="https://user-images.githubusercontent.com/58386334/180709146-79bd1d08-a3f5-4b3f-a503-93327ff674de.png">
그림을 통해 SGD보다 모멘텀의 경로가 '지그재그 정도'가 덜함을 알 수 있음
### 6.1.5 AdaGrad
신경망 학습에서는 학습률($\eta$) 값이 중요함
- 학습률이 너무 작으면 학습 시간이 길어지고,
- 학습률이 너무 크면 발산하여 학습이 제대로 이뤄지지 않음

학습률을 정하는 효과적 기술로 <strong>학습률 감소</strong>가 있음
- 학습률 감소는 학습을 진행하면서 학습률을 점차 줄여가는 방법임
- 이를 더욱 발전 시킨것이 AdaGrad 임

AdaGrad는 개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행함\
AdaGrad는 수식으로 다음과 같이 표현됨\

$$
\begin{aligned}
h\leftarrow h+\frac{dL}{dW}\odot\frac{dL}{dW}[식6.5]\\
W\leftarrow W+\eta \frac{1}{\sqrt{h}}\frac{dL}{dW}[식6.6]
\end{aligned}
$$

($\odot$은 행렬의 원소별 곱셈을 뜻함)\
$h$는 [식 6.5]에서와 같이 기울기의 값을 제곱하여 계속 더해줌,\
그 후, 학습률에 $\frac{1}{\sqrt{h}}$를 곱해 학습률을 조정해준다.\
이는 매개변수의 원소 중에서 많이 움직인 원소는 학습률이 낮아진다는 뜻임\
- AdaGrad는 과기의 기울기를 제곱하여 계속 더하므로, 학습이 진행될수록 갱신 강도가 약해짐
- 무한히 학습한다면 어느 순간 갱신량이 0이 됨
- 이를 개선한 RMSProp라는 방법이 있음
    - RMSProp는 과거 기울기의 반영규모를 기하급수적으로 감소시켜 보완 함

AdaGrad의 구현은 다음과 같음
```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
    
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```
self.h[key]가 0이더라도 0으로 나누는 상황을 방지하기 위해 1e-7을 더해준다\
AdaGrad를 이용하여 [식 6.2]의 최적화 문제를 풀면 다음과 같이 결과가 나온다
<img width="978" alt="img6-6" src="https://user-images.githubusercontent.com/58386334/180709202-b32c0417-3be7-4ce2-bb34-a0a5c24e4c9e.png">
위 그림을 보면 y축은 기울기가 커서 크게 움직이지만, 큰 움직임에 비례해 갱신 정도도 큰 폭으로 작아진다. 그래서 y축 방향으로 갱신 강도가 빠르게 약해지고, 지그재그 움직임이 줄어든다.

### 6.1.6 Adam
Adam은 모멘텀과 AdaGrad 두 기법을 융합하여 생각한 기법이다\
따라서 매개변수 공간을 효율적으로 탐색해줄 것을 기대할 수 있고,\
하이퍼파라미터의 '편향 보정'이 진행된다는 점도 Adam의 특징이다.\
Adam을 이용하여 [식 6.2]의 최적화 문제를 풀어보면 다음과 같은 결과를 얻을 수 있다
<img width="908" alt="img6-7" src="https://user-images.githubusercontent.com/58386334/180709229-ddadd220-f9f5-40ac-b720-959e41ae98c8.png">

### 6.1.7 어느 갱신 방법을 이용할 것인가
[식 6.2]에 대해서는 AdaGras 기법이 가장 나아 보임\
어떤 기법이 더 나은 기법인진 학습률과 신경망의 구조에 따라 결과가 달라질 수 있음

## 6.2 가중치의 초깃값
권장 초깃값에 대해서 설명하고, 실험을 통해 실제로 신경망 학습이 신속하게 이뤄지는 모습을 확인
### 6.2.1 초깃값을 0으로 하면?
<strong>가중치 감소</strong> 기법이란?
- 오버피팅을 억제해 범용 성능을 높이는 테크닉
- 가중치 매개변수의 값이 작아지도록 학습하는 방법

가중치의 초깃값을 균일한 값으로 설정 하면 안됨
- 오차역전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문
- 가중치가 일정하면 순전파를 진행할 때 다음 층의 모든 뉴런에 모두 같은 값이 입력됨
- 이는 가중치를 여러개 갖는 의미가 없어지게 함
### 6.2.2 은닉층의 활성화값 분포
가중치의 초깃값에 따라 은닉층 활성화값들이 어떻게 변화하는지 실험하는 절\
활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 흘려보는 코드는 다음과 같음
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100) #1000개의 데이터
node_num = 100 #각 은닉층의 노드 수
hidden_layer_size = 5 # 은닉층이 5개
activations = {} # 이곳에 활성화 결과를 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = np.random.randn(node_num, node_num) * 1 # 여기서 1이 가중치의 표준편차임
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z
```
이 활성화 값 데이터를 히스토그램으로 그리는 코드는 다음과 같음
```python
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    plt.hist(a.flatten, 30, range-(0,1))
plt.show()
```
이를 실행 해보면 다음 히스토그램을 얻을 수 있다
![img6-10](https://user-images.githubusercontent.com/58386334/180709333-d67896b1-232d-4563-b083-604bc91c9287.jpeg)
각 층의 활성화 값이 0과 1에 치워쳐 분포됨을 알 수 있음\
사용한 활성화 함수인 시그모이드 함수는 출력이 0 또는 1로 가까워지면 미분은 0에 다가간다.\
따라서 데이터가 0과 1에 치우쳐 분포하면 역전파의 기울기 값이 점점 작아져 사라짐\
이는 <strong>기울기 소실</strong>이라 알려진 문제임

가중치 표준편차를 0.01로 바꿔 다시 히스토그램을 얻으면 다음과 같음
![img6-11](https://user-images.githubusercontent.com/58386334/180709351-39401c40-adcb-4ccb-9945-29994bdf9ea5.jpeg)
활성화 값이 0.5 부근에 집중되었지만, 0과 1로 치우치진 않아 기울기 소실 문제가 생기진 않는다.\
해당 경우 문제점
- 활성화 값이 치우쳐 있으므로 뉴런을 여러개 둔 의미가 사라짐
- 표현력을 제한한다는 관점에서 문제가 됨

각 층에서 활성화값이 고루 분포해야 신경망 학습이 효율적으로 이루어 짐\
활성화값 들을 광범위하게 분포시킬 수 있는 <strong>Xavier 초깃값</strong>이 일반적인 딥러닝 프레임 워크에 표준적으로 사용되고 있음\
Xavier 초깃값이란 다음과 같은 규칙을 사용하면 된다

$$
\begin{aligned}
앞 계층의 노&드가 n개라면\\
가중치&의 분포를\\
표준편차가 \frac{1}{\sqrt{n}}&인 분포로 초기화
\end{aligned}
$$

Xaiver 초깃값을 사용하여 다시 활성화값 분포를 알아보면
![img6-13](https://user-images.githubusercontent.com/58386334/180709390-e596151b-110a-4b14-bbf5-a3f648a49a5a.jpeg)
와 같이 각 층에 흐르는 데이터가 적당히 퍼져 있으므로,\
시그모이드 함수의 표현력도 제한받지 않고 효율적으로 학습이 이뤄질 것으로 기대됨\
[그림 6-13]에서 오른쪽으로 갈수록 일그러지는데 sigmoid대신 tanh함수를 사용하면 개선된다.\
sigmoid의 원점이 (0,0.5)에서 대칭인 곡선이기 때문에\
원점이 (0,0)인 tanh를 사용하면 개선됨

Xavier 초깃값은 활성화 함수가 선형인것을 전제로 좋은 초깃값임
- sigmoid와 tanh는 대칭이라 중앙부근을 선형함수로 볼 수 있음
### 6.2.3 ReLU를 사용할 때의 가중치 초깃값
활성화 함수를 ReLU를 사용할때의 경우\
ReLU에 특화된 초깃값으로는 <strong>He 초깃값</strong>이 존재함\
He 초깃값은 앞 계층의 노드가 $n$개일 때, 표준편차가 $\sqrt{\frac{2}{n}}$인 정규 분포를 사용한다.\
표준편차가 0.01인 정규분포, Xavier 초깃값, He 초깃값을 사용한 활성화 값 분포를 비교해 보면 다음과 같다
![img6-14](https://user-images.githubusercontent.com/58386334/180709413-553871dc-b5b8-456a-ab7f-3952c284e496.jpeg)
std=0.01의 히스토그램을 보면 신경망에 아주 작은 데이터가 흐르므로 역전파 때 기울기 역시 작아짐을 알 수 있다. 실제로 학습이 거의 이뤄지지 않을 것으로 보임\
Xavier 초깃값의 히스토그램을 보면 층이 깊어지면서 치우침이 커짐을 확인할 수 있다. 따라서 학습할 때 기울기 소실의 문제가 발생함\
He 초깃값은 모든 층에서 균일하게 분포되어 층이 깊어져도 균일하게 유지되어 역전파에도 적절한 값이 나올 것으로 기대할 수 있음
## 6.3 배치 정규화
배치 정규화란 각 층이 활성화 값을 적당히 퍼뜨리도록 '강제'하자는 아이디어에서 출발한 방법
### 6.3.1 배치 정규화 알고리즘
배치 정규화가 주목받는 이유
- 학습을 빨리 진행할 수 있다
- 초기값에 크게 의존하지 않는다
- 오버피팅을 억제한다

배치 정규화를 활성화 함수의 앞에 삽입함으로써 데이터가 덜 치우치게 할 수 있음\
다음 그림 같이 배치 정규화 계층을 신경망에 삽입함
![img6-16](https://user-images.githubusercontent.com/58386334/180709436-d1aaa594-4f0d-46df-9ebf-3fc25e19b05e.jpeg)
배치 정규화는 미니배치라는 m개의 입력 데이터 집합에 평균이 0, 분산이 1이 되도록 정규화 한다.\
수식은 다음과 같음

$$
\begin{aligned}
\mu &\leftarrow \frac{1}{m} \sum_1^m x_i\\
\sigma_B^2&\leftarrow \frac{1}{m}\sum_1^m (x_i-\mu_B)^2\\
\hat{x_i} &\leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
\end{aligned}
$$

여기서 $\epsilon$은 매우 작은값(10e-7 같이)임 (0으로 나누는 일을 방지)

배치 정규화 계층마다 정규화된 데이터에 고유한 확대와 이동변환을 수행함\
아래 수식과 같음

$$
y_i \leftarrow \gamma \hat{x_i} + \beta
$$

$\gamma$는 확대, $\beta$는 이동을 담당함\
두 값은 처음에는 $\gamma=1,\beta=0$부터 시작하고, 학습하면서 적합한 값으로 조정함\
### 6.3.2 배치 정규화의 효과
- 학습 속도가 빨라짐
- 가중치 초기값에 의존적이지 않고 학습이 안정적으로 진행됨

## 6.4 바른학습을 위해
<strong>오버피팅</strong>이란 신경망이 훈련 데이터에만 지나치게 적응되어 그 외의 데이터에는 제대로 대응하지 못하는 상태를 뜻함
### 6.4.1 오버피팅
오버피팅은 주로 다음의 두 경우에 일어남
- 매개변수가 많고 표현력이 높은 모델
- 훈련 데이터가 적은 경우

오버피팅된 모델의 에폭별 정확도 추이의 그래프 예시는 다음과 같다
![img6-20](https://user-images.githubusercontent.com/58386334/180709461-820b5f04-b81e-49f4-9ced-e56f0b32ff8e.jpeg)
오버피팅된 모델은 훈련에서 사용하지 않은 범용 데이터에 대해 제대로 대응하지 못하는 것을 확인할 수 있음
### 6.4.2 가중치 감소
<strong>가중치 감소</strong>란 학습 과정에서 큰 가중치에 대해서는 큰 페널티를 부과하여 오버피팅를 억제하는 방법이다\
신경망 학습의 목적은 손실함수의 값을 줄이는 것이었음\
이때 가중치 감소를 이루기 위해 가중치의 제곱 노름(L2 노름, $\frac{1}{2}\lambda W^2$)을 손실함수에 더해주는 방법을 사용함\
여기서 $\lambda$는 정규화의 세기를 조절하는 하이퍼 파라미터임\
따라서 $\lambda$를 크게 설정할수록 큰 페널티가 커짐

또한 손실 함수에 $\frac{1}{2}\lambda W^2$를 더해주므로, 가중치의 기울기를 구하는 계산에서는 그동안의 오차역전파법에 따른 결과에 정규화항을 미분한 $\lambda W$를 더해줌

위의 오버피팅된 모델에서 $\lambda=0.1$로 가중치 감소를 적용하면 아래의 그래프와 같아진다.
![img6-21](https://user-images.githubusercontent.com/58386334/180709493-7204f022-93cd-48b4-b8fd-4152b4dac67e.jpeg)
훈련 데이터와 시험 데이터에서의 정확도는 아직 차이가 존재하지만, 가중치 감소를 이용하여 그 차이가 줄어들었음
### 6.4.3 드롭아웃
앞에서 이용한 가중치 감소방법은
- 간단하게 구현이 가능하고 어느정도 지나친 학습을 억제할 수 있음
- 하지만 신경망 모델이 복잡해지면 가중치 감소만으로는 대응하지 어려워짐

이런 경우에는 <strong>드롭아웃</strong>이라는 기법을 사용함

드롭아웃은 뉴런을 임의로 삭제하면서 학습하는 방법임\
삭제된 뉴런은 신호를 전달하지 않음
- 훈련 때는 데이터를 흘릴 때마다 삭제할 뉴런을 무작위로 선택하고
- 시험 때는 모든 뉴런에 신호를 전달함
    - 시험 때는 각 뉴런의 출력에 훈련 때 삭제 안 한 비율을 곱하여 출력함

![img6-22](https://user-images.githubusercontent.com/58386334/180709525-b593ed10-d6a5-4a68-93a0-fdc7ffc90722.jpeg)
드롭아웃은 다음과 같이 구현할 수 있음
```python
class Dropout:
    def __init__*self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
```

## 6.5 적절한 하이퍼파라미터 값 찾기
신경망에서 등장하는 하이퍼파라미터로는 뉴런 수, 배치 크기, 매개변수 갱신 시의 학습률과 가중치 감소 등이 있음
### 6.5.1 검증 데이터
하이퍼파라미터의 성능을 평가할 때는 시험 데이터를 사용해서는 안된다\
시험 데이터를 사용하여 하이퍼파라미터를 조정하면 하이퍼파라미터 값이 시험 데이터에 오버피팅 되기 떄문임\
따라서 하이퍼파라미터를 조정할 때는 하이퍼파라미터 전용 확인 데이터가 필요함\
이 하이퍼파라미터 조정용 데이터를 일반적으로 <strong>검증 데이터</strong>라고 부름
```
훈련 데이터: 매개변수 학습
검증 데이터: 하이퍼파라미터 성능 평가
시험 데이터: 신경망의 범용 성능 평가
```
### 6.5.2 하이퍼파라미터 최적화
하이퍼파라미터 최적화의 핵심은 하이퍼파라미터의 '최적 값'이 존재하는 범위를 조금씩 줄여나간다는 것임\
범위를 줄이려면
- 우선 대략적인 범위를 설정하고
- 그 범위에서 무작위로 하이퍼파라미터 값을 골라낸(샘플링) 후
- 그값으로 정확도를 평가함
- 위 작업을 반복하여 하이퍼파라미터의 '최적 값' 범위를 좁혀나감
    - 샘플링을 할 때 규칙적인 탐색보다는 무작위로 탐색하는 것이 좋은 결과를 낸다고 알려져 있음 (최종 정확도에 미치는 영향력이 하이퍼 파라미터마다 다르기 때문)

하이퍼파라미터의 범위는 '대략적으로' 지정하는 것이 효과적임\
실제로도 0.001에서 1000 사이와 같이 10의 거듭제곱 단위로 지정함 (로그 스케일)

하이퍼 파라미터의 최적화 과정

0. 하이퍼파라미터 값의 범위를 설정함
1. 설정된 범위에서 하이퍼파라미터의 값을 무작위로 추출함
2. 1단계에서 샘플링한 하이퍼파라미터의 값을 사용하여 학습하고, 검증 데이터로 정확도를 평가함 (에폭은 작게 설정함)
3. 1단계와 2단계를 특정 횟수 반복하며, 그 정확도를 보고 하이퍼파라미터의 범위를 좁힘

위 방법을 반복하여 어느 정도 좁아지면 압축한 범위에서 값을 하나 골라냄
- 더 세련된 방법으로 베이즈 최적화가 존재함
## 6.6 정리
이번 장에서 배운 내용
- 매개변수 갱신 방법에는 확률적 경사 하강법(SGD) 외에도 모멘텀, AdaGrad, Adam 등이 있다.
- 가중치 초깃값을 정하는 방법은 올바른 학습을 하는 데 매우 중요하다.
- 가중치의 초깃값으로는 'Xavier 초깃값'과 'He 초깃값'이 효과적이다.
- 배치 정규화를 이용하면 학습을 빠르게 진행할 수 있으며, 초깃값에 영향을 덜 받게 된다.
- 오버피팅을 억제하는 정규화 기술로는 가중치 감소와 드롭아웃이 있다.
- 하이퍼파라미터 값 탐색은 최적 값이 존재할 법한 범위를 점차 좁히면서 하는 것이 효과적이다.
