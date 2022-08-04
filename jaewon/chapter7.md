# 7장 합성곱 신경망(CNN)
CNN은 이미지 인식과 음성인식 등 다양한곳에서 사용됨
## 7.1 전체구조
지금까지 본 신경망은 완전연결 신경망이었음\
CNN에서는 다음 두 계층이 추가됨
- 합성곱 계층 (Conv)
- 풀링 계층 (Pooling)
## 7.2 합성곱 계층
CNN은 각 계층 사이에는 3차원 데이터같이 입체적인 데이터가 흐른다는 점에서 완전연결 신경망과 다름
### 7.2.1 완전연결 계층의 문제점
완전연결 계층의 문제점은 다음과 같음
- 데이터의 형상이 무시됨
    - 이미지와 같은 3차원 형상의 경우는 공간적 정보가 담겨있음
    - 완전연결 계층의 경우 형상을 무시하고 모든 입력 데이터를 동등한 뉴런으로 취급하여 형상에 담긴 정보를 살릴 수 없음

합성곱 계층은 형상을 유지하므로 형상을 가진 데이터를 제대로 이해할 가능성이 있음\
CNN에서는 입출력 데이터를 <strong>특징 맵</strong>이라 함\
합성곱의 입력 데이터는 <strong>입력 특징 맵</strong>, 출력 데이터는 <strong>출력 특징 맵</strong>이라 함

### 7.2.2 합성곱 연산
합성곱 연산에서는 합성곱 연산을 처리함\
합성곱 연산은 이미지 처리의 <strong>필터 연산</strong>에 해당함\
문헌에 따라 필터를 <strong>커널</strong>이라 칭하기도 함

합성곱 연산은 필터의 <strong>윈도우</strong>를 일정 간격으로 이동해가며 입력 데이터에 적용함
- 입력 데이터에서 윈도우가 위치한 부분과 필터를 입력과 필터에서 대응하는 원소끼리 곱한 후 그 총합을 구함
- 그 결과를 해당 장소에 저장함
- 모든 장소에서 수행할 때까지 반복함

CNN에서는 필터의 매개변수가 '가중치'에 해당함\
편향은 필터를 적용한 후 데이터에 적용하여줌
### 7.2.3 패딩
합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정값으로 채우는것을 <strong>패딩</strong>이라 함
```
패딩은 주로 출력의 크기를 조정할 목적으로 사용함
```
### 7.2.4 스트라이드
필터를 적용하는 위치의 간격을 <strong>스트라이드</strong>라 함\
(윈도우가 이동하는 칸 수)\
입력 크기가 $(H,W)$, 필터 크기가 $(FH,FW)$, 출력 크기가$(OH,OW)$, 패딩이 $P$, 스트라이드가 $S$일 떄\
출력 크기는 다음과 같음

$$
OH = \frac{H+2P-FH}{S}+1\\
OW = \frac{W+2P-FW}{S}+1
$$

$\frac{H+2P-FH}{S}$와 $\frac{W+2P-FW}{S}$가 나누어 떨어지는 값이어야 함\
딥러닝 프레임워크 중에는 나누어 떨어지지 않는 경우 반올림 하는 경우도 있음
### 7.2.5 3차원 데이터의 합성곱 연산
3차원 데이터는 세로,가로에 채널까지 고려한 데이터임\
채널쪽으로 특징 맵이 여러개 있다면 입력 데이터와 필터의 합성곱을 채널마다 수행하고, 그 결과를 더해서 하나의 출력을 얻음
- 입력 데이터의 채널 수와 필터의 채널 수가 같아야한다는 점을 주의해야함
### 7.2.6 블록으로 생각하기
3차원 합성곱 연산은 데이터와 필터를 직육면체 블록이라 생각하면 쉬움\
또, 3차원 데이터를 배열로 나타낼 때는 (채널, 높이, 너비)순서로 씀\
출력 데이터를 다수의 채널로 내보내려면 여러 필터를 사용하면 됨\
이 다수의 채널을 모으면 블록이 완성되므로 이 블록을 다음 계층으로 보내는것이 CNN의 처리 흐름임
### 7.2.7 배치 처리
합성곱 연산의 배치처리를 위해 4차원 데이터로 저장하여 연산을 수햄함 (데이터 수, 채널 수, 높이, 너비)
## 7.3 풀링 계층
풀링은 세로,가로 방향의 공간을 줄이는 연산임\
<strong>최대 풀링</strong>은 영역에서의 최댓값을 구하는 연산\
풀링의 윈도우 크기와 스트라이드의 크기를 같게 설정하는 것이 일반적
### 7.3.1 풀링 계층의 특징
- 학습해야 할 매개변수가 없다
- 채널 수가 변하지 않는다
- 입력의 변화에 영향을 적게 받는다
## 7.4 합성곱/풀링 계층 구현하기
풀링 계층은 '트릭'을 사용하면 쉽게 구현할 수 있음
### 7.4.1 4차원 배열
CNN은 4차원 데이터를 다룸
- im2col이라는 '트릭'이 문제를 단순하게 만들어줌
### 7.4.2 im2col로 데이터 전개하기
합성곱 연산을 구현하기위해 for문 대신 im2col이라는 함수를 사용하여 간단하게 구현 가능\
im2col은 입력 데이터를 필터링하기 좋게 전개하는 함수임\
3차원 데이터에 im2col을 적용하면 2차원 행렬로 바뀜

im2col은 필터링하기 좋게 입력 데이터에서 필터를 적용하는 영역을 한줄로 늘어놓음\
이렇게 한줄로 늘어놓는 전개를 필터를 적용하는 모든 영역에서 수행하는 게 im2col임

실제 상황에서는 필터의 적용영역이 겹치는 경우가 대부분임\
이러한 경우 im2col로 전개한 후 원소 수가 원래 블록의 원소 수보다 많아짐
### 7.4.3 합성곱 계층 구현하기
im2col함수의 인터페이스

```python
im2col(input_data, filter_h, filter_w, stride=1, pad=0)
```
input_data - (데이터 수, 채널 수, 높이, 너비)의 4차원 배열로 이루어진 입력 데이터

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T # 필터 전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        return out
```
### 7.4.4 풀링 계층 구현하기
풀링 계층 구현도 합성곱 계층과 마찬가지로 im2col을 사용해 입력 데이터를 전개함\
풀링은 채널쪽이 독립적임
![이미지](https://user-images.githubusercontent.com/58386334/182783040-eddb1136-adf8-4940-b077-9b5d354d212b.jpeg)
```
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 최댓값 (2)
        out = np.max(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
```
풀링 계층은 다음 세 단계로 진행 됨

1. 입력 데이터를 전개한다
2. 행별 최댓값을 구한다
3. 적절한 모양으로 성형한다

## 7.5 CNN 구현하기
```python
class SimpleConvNet:
    def __init__(self, input_dim=(1,28,28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num*(conv_output_size/2) * (conv_output_size/2))
```
위 코드에서는 초기화 인수로 주어진 합성곱 계층의 하이퍼파라미터를 딕셔너리에서 꺼냄\
그리고 합성곱 계층의 출력 크기를 계산함

```python
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
```
위 코드에서는 가중치 매개변수를 초기화 하는 부분임

```python
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
```
위 코드에서는 CNN을 수헝하는 계층들을 생성함\
순서가 있는 딕셔너리인 layers에 계층들을 차례로 추가함

여기까지가 SimpleConvNet의 초기화임

다음의 코드는 추론을 수행하는 predict 매서드와 손실함수의 값을 구하는 loss메서드를 구현함

```python    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
```

아래의 코드는 오차역전파법으로 기울기를 구하는 구현임

```python
    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads
```


## 7.6 CNN 시각화하기
### 7.6.1 1번째 층의 가중치 시각화 하기
![이미지](https://user-images.githubusercontent.com/58386334/182785086-aa3f9405-2a78-4508-a58e-c90d1b6d2a07.jpeg)
학습 전 필터는 무작위로 초기화되고 있어 흑백의 정도에 규칙성이 없음\
학습을 마친 필터는 규칙성이 있는 이미지가 되었음
### 7.6.2 층 깊이에 따른 추출정보 변화
딥러닝 시각화에 관한 연구에 따르면 계층이 깊어질수록 추출되는 정보는 더 추상화 됨

![이미지](https://user-images.githubusercontent.com/58386334/182785348-de52d483-3c45-4bd8-b03a-05fc04805bf2.jpeg)
위와 같은 네트워크 구조는 AlexNet이라 하는데, 합성곱 계층과 풀링 계층을 여러 겹 쌓고, 마지막으로 완전연결 계층을 거쳐 결과를 출력하는 구조임\
딥러닝은 합성곱 계층을 여러 겹 쌓으면, 층이 깊어지면서 더 복잡하고 추상화된 정보가 추출됨\
처음 층은 단순한 에지에 반응하고, 이어서 텍스처에 반응하고, 더 복잡한 사물의 일부에 반응하도록 변화함\
즉, 층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 '고급'정보로 변화해감
## 7.7 대표적인 CNN
### 7.7.1 LeNet
- LeNet은 손글씨를 인식하는 네트워크임
- LeNet은 활성화 함수를 시그모이드 함수를 사용함
- LeNet은 서브샘플링을 하여 중간 데이터의 크기를 줄임
### 7.7.2 AlexNet
구성은 LeNet과 크게 다르지 않음
- 활성화 함수로 ReLU를 이용함
- LRN이라는 국소적 정규화를 실시하는 계층을 이용함
- 드롭아웃을 사용함
## 7.8 정리
- CNN은 지금까지의 완전연결 계층 네트워크에 합성곱 계층과 풀링 계층을 새로 추가함
- 합성곱 계층과 풀링 계층은 im2col 을 이용하면 간단하고 효율적으로 구현할 수 있음
- CNN을 시각화해보면 계층이 깊어질수록 고급 정보가 추출되는 모습을 확인할 수 있음
- 대표적인 CNN에는 LeNet과 AlexNet이 있음
- 딥러닝 발전에는 빅데이터와 GPU가 크게 기여함
