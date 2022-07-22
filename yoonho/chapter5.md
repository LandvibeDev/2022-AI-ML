# Chapter5 오차역전파법
수치 미분은 단순하고 구현하기도 쉽지만 계산 시간이 오래 걸린다   
오차역전파법(backpropagation)

## 5.1 계산 그래프
* 계산 그래프는 계산 과정을 그래프로 나타낸 것
    * 복수의 노드와 에지로 표현

### 5.1.1 계산 그래프로 풀다
<img src="https://user-images.githubusercontent.com/37685575/180347550-f3d7c412-dbb8-4c85-92ac-f9d3c5de15a6.PNG" width=500></img>
<img src="https://user-images.githubusercontent.com/37685575/180347813-b57b6701-e1b5-4f7f-ae97-3c8addea7829.PNG" width=600></img>
<img src="https://user-images.githubusercontent.com/37685575/180347883-00c2c65c-ba94-4cca-9ebe-980f96fd9771.PNG" width=600></img>
* 계산 그래프는 계산 과정을 노드와 화살표로 표현
* 노드는 원으로 표기하고 원 안에 연산 내용을 적음
* 계산 결과를 화살표 위에 적어 각 노드의 계산 결과가 왼쪽에서 오른쪽으로 전해지게 함
* 계산을 왼쪽에서 오른쪽으로 진행하는 단계를 순전파 라고 함
* 왼쪽에서 오른쪽은 역전파

### 5.1.2 국소적 계산
* 계산 그래프는 국소적 계산을 전파함으로써 최종 결과를 얻음
    * 각 노드는 자신과 관련한 계산 외에는 신경 쓸 게없음
    * 단순한 계산을 전달함으로써 복잡한 계산 가능

### 5.1.3 왜 계산 그래프로 푸는가?
* 국소적 계산, 중간 결과 보관
* 가장 큰 이유는 역전파를 통해 미분을 효율적으로 계산 가능
<img src="https://user-images.githubusercontent.com/37685575/180348021-e41add08-edfa-4657-a04b-0e96154a8083.PNG" width=600>

## 5.2 연쇄법칙
### 5.2.1 계산 그래프의 역전파 

<img src="https://user-images.githubusercontent.com/37685575/180348066-cf5de0fa-4efd-47ec-a76b-28aeadec23cf.png" width=400></img>  
* 신호 E에 국소적 미분을 곱한 후 다음 노드로 전달
* 국소적 미분은 f(x)의 미분 값

### 5.2.2 연쇄법칙이란?
* 합성 함수란 여러 함수로 구성된 함수
* 연쇄법칙은 합성 함수의 미분에 대한 성질
    * 합성 함수의 미분은 함성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.   
![ch5_img7](https://user-images.githubusercontent.com/37685575/180348673-565fc054-5c3c-40e6-b9d5-cd58795f8010.png)

### 5.2.3 연쇄법칙과 계산 그래프
![image](https://user-images.githubusercontent.com/37685575/180348874-6f711721-9913-472c-87d6-36cea74fdd18.png)

## 5.3 역전파
### 5.3.1 덧셈 노드의 역전파
![image](https://user-images.githubusercontent.com/37685575/180348928-a841c534-0510-4093-82bf-d1b68962c3d8.png)
* 덧셈 노드의 역전파는 입력된 값을 그대로 다음 노드로 전달

### 5.3.2 곱셈 노드의 역전파
![image](https://user-images.githubusercontent.com/37685575/180349288-0a667b45-d361-4365-8028-b23337fbe7a6.png)
* 곱셈 노드의 역전파는 상류의 값에 순전파 때의 입력 신호들은 서로 바꾼 값을 곱해서 전달

### 5.3.3 사과 쇼핑의 예

## 5.4 단순한 계층 구현하기
* 곱셈 노드를 MulLayer, 덧셈 노드를 AddLayer로 구현

### 5.4.1 곱셈 계층
```Python
class MulLayer
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
```
### 5.4.2 덧셈 계층
```Python
class AddLayer
    def __init__(self):
        pass    # 초기화가 필요없음
    
    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```
```Python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()
 
# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)
 
# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)  # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)  # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # (1)

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650
```

## 5.5 활성화 함수 계층 구현하기
* 신경망을 구성하는 계층 각각을 클래스 하나로 구현
    * ReLU와 Sigmoid 계층 구현

### 5.5.1 ReLU 계층
* ReLU의 수식
![image](https://user-images.githubusercontent.com/37685575/180364044-3e3e176c-758e-479d-8ed4-8d7bdfd27b71.png)
* 미분
![image](https://user-images.githubusercontent.com/37685575/180364102-f78fb551-2d38-4fc3-9fdd-53c6115b437c.png)
    * 순전파 때 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘림
    * 순전파 때 x가 0 이하면 역전파 때는 하류로 신호를 보내지 않음(0을 보냄)
![image](https://user-images.githubusercontent.com/37685575/180364347-9b8af29d-904a-480d-aab1-4c48eae66908.png)
```Python
class Relu:
    def __init__(self):
        self.mask = None
 
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
 
        return out
 
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
 
        return dx
```
### 5.5.2 Sigmoid 계층
* 시그모이드 함수
![image](https://user-images.githubusercontent.com/37685575/180366917-3e93948f-af55-4e52-b838-71ed25e238fa.png)
![image](https://user-images.githubusercontent.com/37685575/180367022-c4ce0a23-a8b3-469e-ae68-90f436667f99.png)
* exp 노드는 y=exp(x), / 노드는 y=1/x 수행
* 역전파 1단계에서 미분 값이 -y^2인 이유는 x에 대한 미분 값이 아니라 / 노드의 입력 값인 1+exp(-x) 에 대한 미분 값이기 때문
* 시그모이드 계층의 역전파는 순전파의 출력(y)만으로 계산할 수 있다.
```Python
class Sigmoid:
    def __init__(self):
        self.out = None
 
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
 
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
 
        return dx
```
## 5.6 Affine/Softmax 계층 구현하기
### 5.6.1 Affine 계층
* 신경망의 순전파 때 수행하는 행렬의 곱은 기하학에서는 어파인 변환(affine transformation)이라고 함
    * 어파인 변환을 수행하는 처리를 Affine 계층으로 구현
![image](https://user-images.githubusercontent.com/37685575/180376080-7c2ee0ba-333a-4222-b464-f8da16423bbe.png)
* 전치행렬이 나오는 이유 : http://taewan.kim/post/backpropagation_matrix_transpose/

### 5.6.2 배치용 Affine 계층
* 데이터 N개를 묶어 순전파 하는경우
![image](https://user-images.githubusercontent.com/37685575/180379250-6ede3d64-52d3-4b00-bce5-f8c60d24b69a.png)
* X의 형상이 (N,2)로 바뀜
* 편향을 더할 때 주의
    * 순전파 때는 각 데이터에 편향이 더해짐
    * 역전파 때는 각 데이터의 역전파 값이 편향의 원소에 모여야 함(0번째 축에 대해서 의 총합)
```Python
class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None
 
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b
 
    return out
 
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
 
    return dx
```
### 5.6.3 Softmax-with-Loss 계층
* Softmax 계층은 입력 값을 정규화 (출력 합이 1이 되도록 변형) 하여 출력
* 손실 함수인 교차 엔트로피 오차도 포함하여 Softmax-with-Loss 계층 구현
![image](https://user-images.githubusercontent.com/37685575/180381967-3530611c-dca8-46a4-9afd-e9cac25e9d95.png)
```Python
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실
        self.y = None # softmax의 출력
        self.t = None # 정답레이블 (원-핫 벡터)
 
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
 
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size 

        return dx
```
## 5.7 오차역전파법 구현하기
### 5.7.1 신경망 학습의 전체 그림
* 오차역전파법은 기울기 산출을 효율적이고 빠르게 구하기 위함

### 5.7.2 오차역전파법을 적용한 신경망 구현하기
* 신경망의 계층을 OrderedDict에 보관함
    * 순전파 때는 추가한 순서대로 각 계층의 forward() 메서드 호출
    * 역전파 때는 계층을 반대로 호출
```Python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
 
 
class TwoLayerNet:
 
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)
 
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
 
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력데이터, t : 정답레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력데이터, t : 정답레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)
 
        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
 
        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
 
        return grads
```

### 5.7.3 오차역전파법으로 구한 기울기 검증하기
* 수치 미분 대신 오차역전파법 사용
* 수치 미분과 오차역전파법의 결과를 비교해서 기울기가 일치하는지 확인 (기울기 확인)
```Python
import sys, os
sys.path.append(os,pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
 
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
 
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
 
x_batch = x_train[:3]
t_batch = t_train[:3]
 
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)
 
# 각 가중치 차이의 절댓값을 구한 후, 그 절댓값들의 평균을 낸다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
```
### 5.7.4 오차역전파법을 사용한 학습 구현하기
```Python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
 
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
 
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
 
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
 
train_loss_list = []
train_acc_list = []
test_acc_list = []
 
iter_per_epoch = max(train_size / batch_size, 1)
 
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 오차역전파법으로 기울기를 구한다
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
```

## 5.8 정리
* 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있다.
* 계산 그래프의 노드는 국소적 계산으로 구성된다. 국소적 계산을 조합해 전체 계산을 구성한다.
* 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분을 구할 수 있다.
* 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있다.
* 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 없는지 확인할 수 있다
* 동작을 계층으로 모듈화 한 덕분에 신경망의 계층을 자유롭게 조합하여 원하는 신경망을 쉽게 만들 수 있다. 


