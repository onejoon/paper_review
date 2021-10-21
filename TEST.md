---
description: >-
  Peter Cho-Ho Lam, Lingyang Chu et al. / Finding Representative Interpretations
  on Convolutional Neural Networks / ICCV 2021
---

# Finding Representative Interpretations on Convolutional Neural Networks \[Kor]

한국어로 쓰인 리뷰를 읽으려면 여기를 누르세요.

## 1. Problem definition

* Despite the success of deep learning models on various tasks, there is a lack of interpretability to understand the decision logic behind deep convolutional neural networks (CNNs).
* It is important to develop representative interpretations of a CNN to reveal the common semantics data that contribute to many closely related predictions.
* How can we find such representative interpretations of a trained CNN?

### Setting

* Consider image classification using CNNs with RELU activation functions
* $$\cal{X}$$: the space of images
* $$C$$: the number of classes
* $$F:\mathcal{X}\rightarrow\mathbb{R}^C$$: a trained CNN, and $$Class(x)=\argmax\_iF\_i(x)$$
* a set of reference images $$R\subseteq\mathcal{X}$$
* $$\psi(x)$$: the feature map produced by the last convolutional layer of $$F$$
* $$\Omega={\psi(x);|;x\in\mathcal{X} }$$ the space of feature maps
* $$G:\Omega\rightarrow\mathbb{R}^C$$, the mapping from the feature map $$\psi(x)$$ to $$Class(x)$$
* $$\mathcal{P}$$: the set of the linear boundaries (hyperplanes) of $$G$$

### Problem

* Finding representative interpretations
  * to find a subset of the linear boundaries $$P(x)\subseteq\mathcal{P}$$ with the largest representativeness
  *   Condition 1: maximize the representativeness of $$P(x)$$

      → maximize $$|P(x)\cap R|$$
  *   Condition 2: avoid covering images in different classes

      → $$|P(x)\cap D(x)|=0$$ where $$D(x)=\{x'\in R\;|\;Class(x')\neq Class(x)\}$$
* Co-clustering problem

$$\max_{P(x)\subseteq\mathcal{P}}|P(x)\cap R|$$



&#x20;$$\max|PR|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|=0$$





\## Motivation ### Related Work ### Idea \*- Find the linear decision boundaries of the convex polytopes that encode the decision logic of a trained CNN \*- Convert the co-clustering problem into a submodular cost submodular cover (SCSC) problem ## Method ### Submodular Cost Submodular Cover problem - SCSC problem $$\max_{P(x)\subseteq\mathcal{Q}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|\leq\delta$$ \*- sample a subset of linear boundaries $$\cal Q$$ from $$\cal P$$ \*- due to sampling, the images covered in the same convex polytope may not be predicted by $$F$$ as the same class → $$\delta$$ \*- We can apply a greedy algorithm for a SCSC problem. ![Untitled](\[Review]%20Finding%20Representative%20Interpretations%20on%20cbb5f8a3e3c94badb112bb7164bafb3a/Untitled%201.png) ### Ranking Similar Images \*- Semantic distance $$Dist(x.x')=\sum_{\mathbf{h}\in P(x)}\Big\vert \langle \overrightarrow{W}_\mathbf{h},\psi(x)\rangle -\langle \overrightarrow{W}_\mathbf{h},\psi(x')\rangle \Big\vert$$ \*- $$\overrightarrow{W}_\mathbf{h}$$ is the normal vector of the hyperplane of a linear boundary $$\mathbf{h}\in P(x)$$ - rank the images covered by $$P(x)$$ according to their semantic distance to $$x$$ in ascending order ## Experiments ![Untitled](\[Review]%20Finding%20Representative%20Interpretations%20on%20cbb5f8a3e3c94badb112bb7164bafb3a/Untitled%202.png) ---
