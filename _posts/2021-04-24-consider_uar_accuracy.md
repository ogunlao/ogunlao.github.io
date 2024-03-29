---
layout: post
title:  "Consider using UAR instead of Accuracy for Imbalanced Classification tasks"
categories: blog
tags: [metrics]
comments: true
# categories: coding
# tags: linux

---

Accuracy is one  of the most used metrics to evaluate classification tasks in machine learning. It is the ratio of the number of correct predictions to the total number of examples. It is simple to understand and compute, which makes it an easy evaluation metric to optimize but it has its shortcomings. Many other metrics such as precision, recall, F1-score etc have majorly been used to sidestep its limitations, but I will like to argue for the Unweighted Average Recall as a good or even better metric to optimize when the sample class ratio is imbalanced, and it is closely related to the accuracy.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

Given a set of samples $\mathcal{X}$ with corresponding labels $\mathcal{Y}$. Let us assume we have a binary classification task with only two labels, $y_1$ and $y_2$.

We can train a classifier using binary cross-entropy loss, hinge loss (or whatever loss is fit) to get the best model for our task. Afterwards, we evaluate this model on unseen data to determine how well it generalizes. This is commonly done by calculating a score like accuracy.

## Decomposing the accuracy score

Accuracy can be computed from the confusion matrix, which gives a breakdown of prediction scores such as true positive $(tp)$, true negative $(tn)$, false positive $(fp)$ and false negative $(fn)$. The goal of this article is not to explain these terms, since there are other great articles online on the subject of confusion matrix.

\begin{equation}
accuracy = \dfrac{total~correct~predictions}{total~number~of~predictions}
\end{equation}

This can be shown to correspond the following from the confusion matrix;
\begin{equation}
accuracy = \dfrac{tp + tn}{tp + tn + fp + fn}
\end{equation}
\begin{equation}
accuracy = \dfrac{tp + tn}{p + n}
\end{equation}
where $p$ is the total no. of positives and $n$ is the total no. of negatives.

The class label we use as positive or negative is arbitrary here. We can further decompose the equation above into two parts;
\begin{equation}
accuracy = \dfrac{tp}{p + n} + \dfrac{tn}{p + n}
\end{equation}

Multiplying the first part by $p$ and second part by $n$ clearly becomes;
\begin{equation}
accuracy = \dfrac{tp}{p}.\dfrac{p}{p + n} + \dfrac{tn}{n}.\dfrac{n}{p + n}
\end{equation}

Let's take the individual elements of the last equation to motivate the UAR.

- $\dfrac{tp}{p}$ is known as the Recall on the positive class and it is the ratio of the total correctly predicted positives to the total number of positives. It is also known as Sensitivity
- $\dfrac{tn}{n}$ is known as the Recall on the negative class and it is the ratio of the total correctly predicted negative to the total number of negatives. It is also known as Specificity

The accuracy score can then be written as follows;

\begin{equation}
accuracy = Sensitivity.\dfrac{p}{p + n} + Specificity.\dfrac{n}{p + n}
\end{equation}

$\dfrac{p}{p + n}$ and $\dfrac{n}{p + n}$ are weights applied to the sensitivity and specificity and both sum to 1. These weights apply a higher score to the recall with more class samples and lower to the other, so it does not weigh the two classes equally. This makes it generally unfit for understanding how well the model is performing for very skewed datasets.

## Balanced Classification Accuracy or Unweighted Average Recall

To mitigate the bias in weighting, we can simply replace the weights with 0.5 or $\dfrac{1}{no~of~classes}$ for the multiclass scenario.

The balanced accuracy then becomes;

\begin{equation}
accuracy_{bal} = Sensitivity\times 0.5 + Specificity\times 0.5
\end{equation}

This balanced accuracy is known as the Unweighted Average Recall, the average of the recall on the positive class and recall on the negative class. There is a correlation between the accuracy and UAR but the UAR gives the correct expectation on class predictions.

## How this is computed

This can be computed from a confusion matrix. Below is a function that computes specificity, sensitivity, accuracy and uar in python given the confusion matrix.

```python
def compute_metrics(confusion_matrix):
    """Calculates specificity, sensitivity, accuracy and uar from confusion matrix
    Confusion matrix of form [[tp, fp]
                              [fn, tn]]
    args:
      confusion_matrix: 2 by 2 nd-array
      output: tuple of float (specificity, sensitivity, accuracy, uar)
    """
    cm = confusion_matrix
    tp, tn = cm[0, 0], cm[1, 1]
    fn, fp = cm[1, 0], cm[0, 1]
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    
    uar = (specificity + sensitivity)/2.0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    metrics_dict = dict(sensitivity=sensitivity, specificity=specificity, 
                       accuracy=accuracy, uar=uar)
    return metrics_dict
```

Let us look at some cases of how accuracy and UAR differ in their metrics

**Case 1: Balanced class**
Here, we have 50 samples from the positive class and 50 samples from the negative class. The model predicts 45 of 50 samples correctly for the positive class and 39 of 50 samples correctly for the negative class.

```python
>> import numpy as np
>> cm = np.array([[45, 11],
                  [5, 39]])
>> print(compute_metrics(cm))
```

```python
# Note that values have been rounded up to 2 decimal places
output: {'sensitivity': 0.90, 'specificity': 0.78, 'accuracy': 0.84, 'uar': 0.84}
```

Since the classes are balanced, the UAR and accuracy both give similar information

**Case 2: Imbalanced class**
In this case, we have 20 samples from the positive class and 80 samples from negative class. The model predicts 4 of 20 samples correctly for the positive class and 75 of 80 samples correcly for the negative class.

```python
>> cm = np.array([[4, 5],
                  [16, 75]])
>> print(compute_metrics(cm))
```

```python
output: {'sensitivity': 0.20, 'specificity': 0.94, 'accuracy': 0.79, 'uar': 0.57}
```

Here, the accuracy is over optimistic and indicating that we are doing well on about $79 \%$ of the samples. This is because of the size of samples of the negative class. However, the UAR gives a truer result showing that we still have work to do. Looking at the specificity and sensitivity closely, they show where the problem lies. The model is doing poorly on the positive class \(as indicated by the sensitivity, $20 \%$\)

You can play with different formulations of the confusion matrix to better understand how class imbalance affects the scores.

> I first saw the UAR score on [Charles Onu](https://onucharles.github.io/)'s Master thesis in the section "Balanced Classification Accuracy and Misclassification Loss". This is a decomposition of the equation.

**Update**
Sep 11, 21: Fixed in a bug in computing the UAR in an earlier version of this article, and added clearer examples

## Conclusion

Accuracy is the most used metric for evaluating machine learning classification tasks. In this article, we decomposed the accuracy into individual ratios composed of the sensitivity and specificity weighted by a class ratio. Then, we fixed the bias in the accuracy by giving equal weights to both scores. This led to the Unweighted Average Recall.
