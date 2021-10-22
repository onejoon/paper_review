---
description: >-
  Peter Cho-Ho Lam, Lingyang Chu et al. / Finding Representative Interpretations
  on Convolutional Neural Networks / ICCV 2021
---

# Finding Representative Interpretations on Convolutional Neural Networks \[Eng]

한국어로 쓰인 리뷰를 읽으려면 [여기](RI\_review\_Kor.md)를 누르세요.

## 1. Problem definition

* Despite the success of deep learning models on various tasks, there is a lack of interpretability to understand the decision logic behind deep convolutional neural networks (CNNs).
* It is important to develop representative interpretations of a CNN to reveal the common semantics data that contribute to many closely related predictions.
* How can we find such representative interpretations of a trained CNN?

### Setting

* Consider image classification using CNNs with RELU activation functions
* $$\cal{X}$$: the space of images
* $$C$$: the number of classes
* $$F:\mathcal{X}\rightarrow\mathbb{R}^C$$: a trained CNN, and $$Class(x)=\argmax_i F_i(x)$$
* a set of reference images $$R\subseteq\mathcal{X}$$
* $$\psi(x)$$: the feature map produced by the last convolutional layer of $$F$$
* $$\Omega=\{\psi(x)\;|\;x\in\mathcal{X} \}$$ the space of feature maps
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

$$
\max_{P(x)\subseteq\mathcal{P}}|P(x)\cap R|\\ \mathsf{s.t.}\quad|P(x)\cap D(x)|=0
$$

![Figure 1. Finding the optimal subset of linear boundaries](.gitbook/assets/RI\_cnn\_prob\_def.png)

## 2. Motivation

### Related Work

There are various types of existing interpretation methods for CNNs.

1. Gradient-baed methods
   * Compute and visualize the gradient of the score of a predicted class w.r.t. an input image.
   * Such interpretations are not representative due to high sensitivity to the input noise.
2. Model approximation methods
   * Approximate a deep neural network locally or globally with an interpretable agent model.
   * Most of those methods perform poorly on modern CNNs trained on complicated data.
3. Conceptual interpretation methods
   * identify a set of concepts that contribute to the predictions on a pre-defined group of conceptually similar images.
   * These methods require sophisticated customization on deep neural networks.
4. Example-based methods
   * Find exemplar images to interpret the decision of a deep neural network.
   * Prototype-based methods summarize the entire model using a small number of instances as prototypes.
   * The selection of prototypes considers little about the decision process of the model.

### Idea

* Find the linear decision boundaries of the convex polytopes that encode the decision logic of a trained CNN
* Convert the co-clustering problem into a submodular cost submodular cover (SCSC) problem

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

### Reviewer

## Reference & Additional materials

***
