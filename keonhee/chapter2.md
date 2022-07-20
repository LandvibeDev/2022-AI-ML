## 1. 퍼셉트론?

- 다수의 신호를 입력으로 받아 하나의 신호를 출력
- 퍼셉트론의 신호: 1 or 0
- 퍼셉트론의 예  
  ![4719AD45-6038-48C3-9C37-F64D7BBE2F96_4_5005_c](https://user-images.githubusercontent.com/16450962/179205337-f3bb1db6-8014-4158-85e1-13931be9aacb.jpeg)
    - 그림의 원을 뉴런 or 노드라고 부름
    - 입력 신호가 뉴런에 보내질 떄에는 각각 고유한 가중치가 곱해짐(w1x1, w2x2)
    - 뉴런에서 보내온 신호의 총합이 정해진 한계를 넘어설 때만 1을 출력
- 수식으로 표현  
  <img width="325" alt="image" src="https://user-images.githubusercontent.com/16450962/179356336-b5ea4b2c-8067-4346-af73-7203f705937d.png">

## 2. 단순한 논리 회로
퍼셉트론을 활용한 간단한 문제 살펴보기

### 2.1 AND 게이트

<img width="318" alt="image" src="https://user-images.githubusercontent.com/16450962/179356576-f4f9265d-1664-4c1a-a4ea-caeb7f337ffb.png">
- 퍼셉트론으로 표현?    

- 위 진리표를 만족하는 (w1, w2, ⍬)  구하기
    - (0.5, 0.5, 0.7)

|  가중치 계산  |  임계점 확인  |  y 값 | 
| --- | --- | --- |
| 0 * 0.5 + 0 * 0.5 = 0   | 0 < 0.7 |  0 |
| 1 * 0.5 + 0 * 0.5 = 0.5 | 0.5 < 0.7|  0 |
| 0 * 0.5 + 1 * 0.5 = 0.5 | 0.5 < 0.7|  0 |
| 1 * 0.5 + 1 * 0.5 = 1 |  1 > 0.7  | 1 |


### 2.2 NAND 게이트와 OR 게이트
<img width="323" alt="image" src="https://user-images.githubusercontent.com/16450962/179356584-854ba76d-c0c7-4a60-9596-dc833ab49638.png">

- 위 진리표를 만족하는 (w1, w2, ⍬)  구하기
    - (-0.5, -0.5, -0.7)
    - AND 게이트의 매개변수 부호를 반전하면 NAND 게이트가 됨

|  가중치 계산  |  임계점 확인  |  y 값 | 
| --- | --- | --- |
| 0 * -0.5 + 0 * -0.5 = 0   | 0  > -0.7 |  1 |
| 1 * -0.5 + 0 * -0.5 = -0.5 | -0.5 > -0.7|  1 |
| 0 * -0.5 + 1 * -0.5 = -0.5 | -0.5 > -0.7|  1 |
| 1 * -0.5 + 1 * -0.5 =  1.0 |  -1 <- 0.7 | 0 |


<img width="315" alt="image" src="https://user-images.githubusercontent.com/16450962/179356589-637c03f4-a342-420b-83a7-ce8ff57ebfa2.png">

- 위 진리표를 만족하는 (w1, w2, ⍬)  구하기
    - (0.5, 0.5, 0.3)

|  가중치 계산  |  임계점 확인  |  y 값 | 
| --- | --- | --- |
| 0 * 0.5 + 0 * 0.5 = 0   | 0 < 0.3 |  0 |
| 1 * 0.5 + 0 * 0.5 = 0.5 | 0.5 > 0.3|  1 |
| 0 * 0.5 + 1 * 0.5 = 0.5 | 0.5 > 0.3|  1 |
| 1 * 0.5 + 1 * 0.5 = 1 |  1 > 0.3  | 1 |


- 퍼셉트론의 구조는 AND, NAND, OR 게이트에서 모두 똑같음

## 3. 퍼셉트론 구현

### 3.1 간단한 구현 / 3.2 가중치와 편향 도입 / 3.3 가중치와 편향 구현하기

```python
def ANDByBiasAndWeight(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```

- -⍬가 편향 b로 치환됨
- 편향b는 가중치 w1, w2와 다름
    - 가중치 w1, w2: 입력 신호가 결과에 주는 영향력(중요도)을 조절
    - 편향 b는 뉴런이 얼마나 수비게 활성화(결과로 1을 출력)하느냐를 조정하는 매개변수

## 4. 퍼셉트론의 한계

### 4.1 XOR 구현하기
<img width="335" alt="image" src="https://user-images.githubusercontent.com/16450962/179357702-2ced3eb6-8709-4e09-9231-0f2ce19a40a1.png">

- 퍼셉트론으로는 XOR 게이트를 구현 불가
- WHY?
    - (b, w1, w2) = (-0.5, 1.0, 1.0) 일떄 위의 진리표를 만족하고 아래의 퍼셉트론 식을 따름
      <img width="257" alt="image" src="https://user-images.githubusercontent.com/16450962/179357788-2344f460-9a8c-4333-ac3d-90f449beb4df.png">

<img width="669" alt="image" src="https://user-images.githubusercontent.com/16450962/179357782-ca02d741-4fdd-4a10-a87f-81ba76748fab.png">

- (1, 1) 일 때 0이여야 하지만 퍼셉트론은 직선으로만 표현 가능하기 때문에 ●와 ▲ 를 나누지 못함

### 4.2 선형과 비선형
<img width="403" alt="image" src="https://user-images.githubusercontent.com/16450962/179357911-e36aaf19-ba17-45af-91a7-651c9ceda1da.png">
- 위 처럼 곡선으로 나눌 수 있음
- 비선형: 그림 2-8과 같은 곡선
- 선형: 그림 2-6과 같은 직선


## 5. 다층 퍼셉트론이 충돌한다면
- 단층 퍼셉트론으로는 XOR 게이트를 표현할 수 없음
- 층을 쌓아 만드는 다층 퍼셉트론으로 XOR 게이트를 표현할 수 있음


### 5.1 기존 게이트 조합하기

<img width="379" alt="image" src="https://user-images.githubusercontent.com/16450962/179357980-54777026-1e45-4aae-844a-9eb0c5d53c60.png">

<img width="425" alt="image" src="https://user-images.githubusercontent.com/16450962/179357992-6265424d-2a65-4724-b47b-4adec7b63196.png">

- AND, NAND, OR 게이트를 조합하여 XOR 게이트를 만들 수 있음

<img width="320" alt="image" src="https://user-images.githubusercontent.com/16450962/179358012-5c031149-b190-4870-863c-5c7bb3c2fdf6.png">

### 5.2 XOR 게이트 구현하기
- 위에서 구현했던 NAND, OR, AND 함수로 구현 가능
```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

<img width="388" alt="image" src="https://user-images.githubusercontent.com/16450962/179358080-3de567b1-5c96-422e-bd2e-593452fb5a05.png">

- XOR는 다층 퍼셉트론 구조 네트워크
- 왼쪽부터 0층, 1층, 2층


## 6. NAND에서 컴퓨터까지

- 다층 퍼셉트론으로 복잡한 회로 구현 가능
    - 덧셈을 처리하는 가산기
    - 2진수를 10진수로 변환하는 인코더
    - 조건을 충족하면 1을 출력하는 페리티 검사 회로
    - 컴퓨터
        - 이론상 NAND 게이트의 조합으로 구현 가능


## 7. 정리
- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다
- 퍼셉트론에서는 '가중치'와 '편향'을 매개변수로 설정한다
- 퍼셉트론으로 AND, OR 게이트 등의 논리회로를 표현할 수 있다
- XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없다
- 2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있다
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다
- 다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있다

