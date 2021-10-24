---
description: >-
  Peter Cho-Ho Lam, Lingyang Chu et al. / Finding Representative Interpretations
  on Convolutional Neural Networks / ICCV 2021
---

# Finding Representative Interpretations on Convolutional Neural Networks \[Kor]

[English version](RI\_review\_Eng.md) of this article is available.

## 1. Problem definition

* 최근 다양한 영역에서 딥러닝 기반의 인공지능 모델들이 성공적인 성능을 보이고 있지만, **Deep convolutional neural networks(CNNs)의 의사결정 과정에 대한 해석은 아직 부족**하다. 이에 대한 충분한 해석성이 제공되어야 딥러닝 모델들을 신뢰가능하게 만들 수 있을 것이다.
* 이 논문에서는 비슷한 예측을 갖는 관련 높은 데이터들을 대표하는 **common semantics를 알아내기 위해 representative interpretations를 찾고자** 한다. 즉, representative interpretations는 CNN의 의사결정 과정에서 구분되는 대표적인 특징 그룹을 보여준다.
* 어떻게 학습된 CNN으로부터 이러한 representative interpretations를 찾을 수 있을까?

### Notation

이미지 분류 문제에서 학습된 ReLU activation function을 사용하는 CNN 모델을 생각해보자.

* $$\cal{X}$$: 이미지 공간
* $$C$$: 이미지 클래스의 수
* $$F:\mathcal{X}\rightarrow\mathbb{R}^C$$: 학습된 CNN, $$Class(x)=\argmax_i F_i(x)$$
* Reference images의 집합 $$R\subseteq\mathcal{X}$$
* $$\psi(x)$$: $$F$$의 마지막 convolutional layer로부터 생성된 feature map
* $$\Omega=\{\psi(x)\;|\;x\in\mathcal{X} \}$$ feature map 공간
* $$G:\Omega\rightarrow\mathbb{R}^C$$, feature map $$\psi(x)$$를 $$Class(x)$$로 매핑하는 함수
* $$\mathcal{P}$$: $$G$$의 linear boundaries(hyperplanes)의 집합

{% hint style="info" %}
* Reference images는 이 방법을 통해 해석하고 싶은 unlabeled images를 가리킨다.
{% endhint %}

### Representaitive Interpretation

문제를 formulation하기 앞서 representative interpretation을 찾는다는 목표구체화할 필요가 있다.

*   \[Representative interpretation]&#x20;

    이미지 $$x\in\mathcal{X}$$에 대한 representative interpretation은 $$x$$에 대한 모델 $$F$$의 일반적인 의사결정을 드러내는 해석을 의미한다.
* 학습된 DNN 모델의 예측을 feature map을 통해 분석할 때, 많은 현존하는 연구에서 마지막 layer로부터 최종 class로의 매핑인 $$G$$를 이용하여 의사결정 로직을 설명한다.

![Decision logic of a CNN](.gitbook/assets/cnn\_decision\_logic.png)

*   \[Linear boundaries]

    $$G$$로 인한 의사결정 과정은 연결 hyperplanes의 조각들로 구성된 piecewise linear decision boundary로 특징지어질 수 있다. $$G$$의 linear boundaries의 집합을 $$\cal{P}$$라 하자.
* $$\cal P$$의 linear boundaries는 feature map space $$\Omega$$를 convex polytopes로 나눈다. 각각의 convex polytope는 해당 지역 안에 있는 이미지들을 동일한 class로 분류하는 decision region을 정의한다.
* 따라서 $$\cal P$$의 부분집합으로부터 $$x$$를 포함한 decision region을 잘 정의하는 것이 representative interpretation을 제공한다. 즉, 좋은 representative interpretation에 대응되는 $$P(x)\subseteq\mathcal{P}$$를 찾는 것이 목표이다.

{% hint style="info" %}
\[Goal]&#x20;

각 image $$x$$에 대하여 좋은 representative interpretation이 될 수 있는 decision region $$P(x)\subseteq\mathcal{P}$$를 찾자.
{% endhint %}



### Finding Representative Interpretations

'좋은' representative interpretations란 무엇일까? 이는 다음과 같은 두가지 조건을 만족해야한다.

1.  $$P(x)$$의 representativeness를 최대화해야 한다.

    \-> Decision region $$P(x)$$가 최대한 많은 reference images를 커버해야한다.

    \-> maximize $$|P(x)\cap R|$$
2.  &#x20;$$x$$와 다른 class에 속하는 이미지들을 포함하지 않아야 한다.

    → $$|P(x)\cap D(x)|=0$$ where $$D(x)=\{x'\in R\;|\;Class(x')\neq Class(x)\}$$

이는 다음과 같은 최적화 문제로 표현할 수 있다.

* Co-clustering problem

$$
\max_{P(x)\subseteq\mathcal{P}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|=0
$$

![Finding the optimal subset of linear boundaries](.gitbook/assets/RI\_cnn\_prob\_def.png)

## 2. Motivation

### Related Work

CNN의 로직을 설명하기 위한 다양한 해석기법들이 연구되어 왔다.

1. Conceptual interpretation methods
   * 컨셉적으로 비슷한 이미지들로 사전에 정의된 그룹에서 예측에 기여하는 컨셉들의 집합을 찾는 방법이다.
   * 그러나 이 방법은 DNN에 복잡한 customization을 요구하기 때문에 일반적인 CNN에 범용적으로 적용하기 어렵다.
2. Example-based methods
   * DNN의 의사결정을 해석하기 위해 모범 이미지(exemplar images)를 찾는다.
   * Prototype-based methods는 prototypes라 불리는 적은 수의 instances를 사용하여 전체 모델을 요약한다.
   * Prototype selection 방법은 모델의 의사결정 과정에 대한 고려가 부족할 수 있다.

### Idea

이 논문은 일반적은 CNN 모델에서 decision boundaries를 고려하여 의사결정에 대한 대표적인 해석성을 제공하는 것을 목표로 한다.

* 학습된 CNN의 decision logic을 encode하여 interpretation을 제공하는 decision region을 찾자.
* 이 문제를 이전 Section에서 co-clustering problem으로 formulation하였다.
* Co-clustering problem을 submodular cost submodular cover(SCSC) problem으로 치환하여 최적화 문제를 풀 수 있도록 제안한다.

## 3. Method

### Submodular Cost Submodular Cover problem

* SCSC problem

$$
\max_{P(x)\subseteq\mathcal{Q}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|\leq\delta
$$

* sample a subset of linear boundaries $$\cal Q$$ from $$\cal P$$
* due to sampling, the images covered in the same convex polytope may not be predicted by $$F$$ as the same class → $$\delta$$ \*- We can apply a greedy algorithm for a SCSC problem. ![Untitled](\[Review]%20Finding%20Representative%20Interpretations%20on%20cbb5f8a3e3c94badb112bb7164bafb3a/Untitled%201.png)

### Ranking Similar Images

*   Semantic distance

    $$
    Dist(x.x')=\sum_{\mathbf{h}\in P(x)}\Big\vert \langle \overrightarrow{W}_\mathbf{h},\psi(x)\rangle -\langle \overrightarrow{W}_\mathbf{h},\psi(x')\rangle \Big\vert
    $$
* $$\overrightarrow{W}_\mathbf{h}$$ is the normal vector of the hyperplane of a linear boundary $$\mathbf{h}\in P(x)$$
* rank the images covered by $$P(x)$$ according to their semantic distance to $$x$$ in ascending order

## 4. Experiment & Result

![Untitled](\[Review]%20Finding%20Representative%20Interpretations%20on%20cbb5f8a3e3c94badb112bb7164bafb3a/Untitled%202.png)

### Experimental setup

### Result

## 5. Conclusion

### Take home message



## Author / Reviewer information

### Author

**장원준 (Wonjoon Chang)**

* KAIST AI
* one\_jj@kaist.ac.kr

### Reviewer

*

## Reference & Additional materials

1. Lam, Peter Cho-Ho, et al. "Finding representative interpretations on convolutional neural networks." _Proceedings of the IEEE/CVF International Conference on Computer Vision_. 2021.
2.

***
