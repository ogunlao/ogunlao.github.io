---
layout: post
title:  "K Nearest Neighbor as a Neural Network"
# categories: article
tags: [softmax, knn, neural_network]
comments: true
# categories: coding
# tags: linux

---

A Neural network is a universal function approximator, so in theory it is possible to learn any function using a neural network. As K-nearest neighbor is a method of predicting the label of a new datapoint from the test set, it is possible to express its prediction function as a neural network, although less intuitive. This article will show how to express a 1-Nearest Neighbor as a neural network, using the datapoints.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

k-Nearest Neighbor uses a distance function to evaluate the label of a new test point. The method of evaluation involves taking the average of predictions of k nearest points to the given test point. It often serves a base model for many predictions tasks and often difficult to beat.

Given a set of train data-points,
\begin{equation}
\{(x_1, y_1), (x_2, y_2), ... ,(x_n, y_n)\}
\end{equation}

with $x_i \in \mathcal{R}^d$, $y_i \in \mathcal{R}$ and i = 1, 2, ..., n. There are $n$ training examples each with d number of features. Given a new datapoint $x_t$, how can we classify the point using k-nearest neighbor into its correct class?

## K Nearest Neighbor

To classify a new test point into its correct label, we perform the following steps:

- Calculate the distance between each datapoint in the training example and the test datapoint
- Assign the test data point the label of the k datapoints with minimum distance. for a classification task, we can take the mode of the k classes, while for regressing task, we average the predictions to get the prediction.
- k can range from 1 to n, but usually between 1 and 10.
- The distance function takes in each $x_i$ and $x^t$ and outputs a scalar. The distance function to use may depend on the task but a popular and common distance is the l2-distance. The l2-distance can be expressed as:

\begin{equation}
d = \sqrt{\sum_{j=1}^{d} (x_{ij} - x^t_j)^2}
\end{equation}
Using the matrix notation, 
\begin{equation}
d = \sqrt{\sum_{j=1}^{d} (X^2_j - 1_n.(x^t)^T_j)^2}
\end{equation}
where $X = (x_1, x_2, ..., x_n)^T \in \mathcal{R}^{nxd}$ and $(x^t)^T \in \mathcal{R}^d$. To simplify notation, I will assume that $x^t$ will be broadcasted along the matrix, which becomes simply:
\begin{equation}
d = \sqrt{\sum_{j=1}^{d} (X^2_j - x^t_j)^2}
\end{equation}

## Expressing KNN as a Neural Network
With the understanding of the distance function, we can break it apart to derive the parameters of our neural network.

From the distance function, 
\begin{equation}
d^2 = \sum_{j=1}^{d} (x_{ij} - x_j)^2 = d'
\end{equation}
Since optimizing $d^2$ is equivalent to optimizing for $d$, we work with $d^2$ instead, which we will call $d'$

### Layer 1: Computing the distance function
Expanding the equation, we get:

\begin{equation}
d' = \sum_{j=1}^{d} (X^2_j - x^t_j) \odot (X^2_j - x^t_j)
\end{equation}
Note that: $\odot$ is a hadamard product, i.e. element-wise product between the two matrices.
\begin{equation}
d' = \sum_{j=1}^{d} (X^2_j + (x^t_j)^2 - 2X_j.x^t
\end{equation}

\begin{equation}
d' = -2X_jx^t + \sum_{j=1}^{d} (X^2_j + (x^t_j)^2
\end{equation}
since the $-2X_jx^t$ does not depend on j.

At this point we can easily extract our first layer, $Z_1 = W_1x_1 + b$ where $W_1 = -2X$, $x_1 = x^t$ and $b = \sum_{j=1}^{d} (X^2_j + (x^t_j)^2$

### Layer 2: Softmax Layer

As we are looking for the class with minimum distance from the test datapoint, we have to negate the vector, so the datapoint with minimum distance, then have the maximum value. We then pass this through a softmax layer to normalize.

\begin{equation}
Z_2 = softmax(-Z_1)
\end{equation}

### Layer 3: Prediction Layer

Before now, we have not really talked out the labels of the training examples. It comes in at this layer to support in prediction. 

- For a regression task, this computation is almost done. We take the vector of distances and find the prediction of the class, with the minimum distance (or maximum value in this case, as we have performed inversion).
- For a classification task, we can also take the label of the datapoint with the minimum distance or go a step further.

For a classification task, where $Z_3 = W_3x_3 + b$, firstly, we perform one-hot encoding on the train labels. $y_{onehot} \in \mathcal{R}^{nxd}$

z_3 = z_2.T@y_train_onehot
$Z_3 = Z_2^Ty_{onehot} + b$ where $W_3 = X^T$, $x_3 = z_2$, $b = 0$

## Implementation

I created a jupyter notebook to show predictions on the iris dataset. You can access the notebook [via this link](\notebooks\knn_as_neural_network.ipynb). Feel free to drop comments, and possibly give area for clarification or improvement. Let me try to explain the major parts of the implementation.

- Data: Loaded the iris dataset via the sklearn load dataset api.
- Preprocessing: Normalized the dataset (often a good thing to normalize) and converted each label into one-hot encoded vectors.
- Built 3 layer neural network using the datapoints as parameters.

## Conclusion

In this article, We showed how a k-nearest neighbor classifier can be transformed into a neural network using the datapoints as parameters. This is also a non-parametric model and the weights of the model increases as the number of training points increases. Finally, neural network is a universal function approximator and can therefore be an exercise to represent other models in terms of a basic neural network model.