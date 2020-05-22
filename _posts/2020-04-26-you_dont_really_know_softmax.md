---
layout: post
title:  "You Don't Really Know Softmax"
categories: article
tags: [softmax, numerical stability]
comments: true
# categories: coding
# tags: linux

---

Softmax function is one of the major functions used in classification models. It is usually introduced early in a machine learning class. It takes as input a real-valued vector of length, d and normalizes it into a probability distribution. It is easy to understand and interprete but at its core are some gotchas than one needs to be aware of. This includes its implementation in practice, numerical stability and applications. The article is an exposé on the topic.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

Softmax is a non-linear function, used majorly at the output of classifiers for multi-class classification. Given a vector $[x_1, x_2, x_3, ... x_d]$ for $i = 1,2, ...d$, the softmax function has the form

\begin{equation}
sm(x_i) = \dfrac{e^x_i}{\sum_{j=1}^{d} e^{x_j}}
\end{equation}

where d is the number of classes.  
The sum of all the exponentiated values, $\sum_{j=1}^{d} e^{x_j}$ is a normalizing constant which helps to ensure that it maintains the properties of a probability distribution i.e. a) the values must sum to 1 b) they must be between 0 and 1 inclusive $[0, 1]$.   

![Softmax classifier](/images/softmax.png "source: ljvmiranda921.github.io")  

For example, given a vector $x = [10, 2, 40, 4]$, to calculate the softmax of each element;   

- exponentiate each value in the vector $e^x = [e^{10}, e^2, e^{40}, e^4]$,  
- calculate the sum $\sum{e^x} = e^{10} + e^2 + e^{40} + e^4 = 2.353...e^{17}$
- then, divide each $x_i$ by the sum to give $sm(x) = [9.35762297e^{-14}, 3.13913279e^{-17}, 1.00000000e^{+00}, 2.31952283e^{-16}]$

This can be easily implemented in a numerical library like numpy,
```
>> import numpy as np
def softmax(x):
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  sm_x = exp_x/sum_exp_x
  return sm_x
   
>> x = np.array([10, 2, 40, 4])
>> print(softmax(x))
output: [9.35762297e-14 3.13913279e-17 1.00000000e+00 2.31952283e-16]
  
```

- Questions
  - What do you observe about the output?
  - Will the output sum to 1?

These are pointers to what we will be discussing in the next sessions?

## Numerical Stability of Softmax
From the softmax probabilities above, we can deduce that softmax can become numerically unstable for values with a very large range. Consider changing the 3rd value in the input vector to $10000$ and re-evaluate the softmax.  

```
>> x = np.array([10, 2, 10000, 4])
>> print(softmax(x))
output: [0.0,  0.0, nan,  0.0]
```
'nan' stands for not-a-number and occurs when there is an overflow or underflow. But, why the $0$s and $nan$? Are we implying we cannot get a probability distribution from the vector? 
- Question: Can you find out what caused the overflow?

Exponentiating a large number like $10000$ leads to a very, very large number. This is approximately $2^{10000}$. This causes overflow.

- Can we do better? Well, we can.
Taking our original equation, 
\begin{equation} 
sm(x_i) = \dfrac{e^x_i}{\sum_{j=1}^{d} e^{x_j}}
\end{equation} 
Let's subtract a constant $c$ from the $x_i$s  
\begin{equation} 
sm(x_i) = \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}}
\end{equation} 
We just shift the $x_i$ by a constant. If this shifting constant, $c$ is the maximum of the vector, $max(x)$, then we can stabilize our softmax computation.

- Question: Do we get the same answer as the original softmax?    
This can be shown to be equivalent to the original softmax function:  
Consider  

\begin{equation}  
sm(x_i) = \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}}
\end{equation}
\begin{equation}
sm(x_i) = \dfrac{e^{x_i}e^{-c}}{\sum_{j=1}^{d} e^{x_j}e^{-c}}
\end{equation} 
\begin{equation}
sm(x_i) = \dfrac{e^{x_i}e^{-c}}{e^{-c}\sum_{j=1}^{d} e^{x_j}}   
\end{equation}  

which produces the same initial softmax  

\begin{equation}
sm(x_i) = \dfrac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}  
\end{equation}  

A numpy implementation of this stable softmax will look like this:
```
def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    sm_x = exp_x/sum_exp_x
    return sm_x
```
if we apply it to our old problem
```  
>> x = np.array([10, 2, 10000, 4])
>> print(softmax(x))
output: [0., 0., 1., 0.]
```
Great, problem solved !!!
- Question: Why are all other values in the softmax 0. Does it mean they have no probability of occuring?

## Log Softmax

A critical evaluation of the softmax computation shows a pattern of exponentiations and divisions. Can we reduce these computations? We can instead optimize the log softmax. This gives us nice characteristics such as;  
1. numerical stability.
1. gradient of log softmax becomes additive since $log(a/b) = log(a) - log(b)$
1. lesser computations of divisions and multiplications as addition is less computationally expensive.
1. log is also a monotonically increasing function. we get this property for free  

To quote a  [stackoverflow answer](https://datascience.stackexchange.com/a/40719) on using log softmax over softmax:
> There are a number of advantages of using log softmax over softmax including practical reasons like improved numerical performance and gradient optimization. These advantages can be extremely important for implementation especially when training a model can be computationally challenging and expensive. At the heart of using log-softmax over softmax is the use of log probabilities over probabilities, which has nice information theoretic interpretations.
When used for classifiers the log-softmax has the effect of heavily penalizing the model when it fails to predict a correct class. Whether or not that penalization works well for solving your problem is open to your testing, so both log-softmax and softmax are worth using.

If we naively apply the logarithm function to the probability distribution, we get
```
>> x = np.array([10, 2, 10000, 4])
>> softmax(x)
output: [0., 0., 1., 0.]
>> np.log(softmax(x))
output: [-inf, -inf,   0., -inf]
```
We are back to numerical instability, in particular, numerical underflow.  
- Question: Why is this so?   
The answer lies in taking the logarithm of individual elements. The $log(0)$ is undefined. Can we do better? oh yes!

## Log-Softmax Derivation

\begin{equation}
sm(x_i) = \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}}
\end{equation}
\begin{equation}
log~sm(x_i) = log \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}}
\end{equation}
\begin{equation}
log~sm(x_i) = x_i - c - log {\sum_{j=1}^{d} e^{x_j -c}}
\end{equation}
- What if we want to get back our original probabilities?
Well, we can exponentiate and normalize the log softmax or log probability values.
\begin{equation}
  sm(x_i) = \dfrac{e^{log~probs}}{\sum_{j=1}^{d} e^{log~probs}}
\end{equation}
Let's make this concrete via code.

```
def logsoftmax(x, recover_probs=True):
    # LogSoftMax Implementation 
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    log_sum_exp_x = np.log(sum_exp_x)
    max_plus_log_sum_exp_x = max_x + log_sum_exp_x
    log_probs = x - max_plus_log_sum_exp_x

    # Recover probs
    if recover_probs:
        exp_log_probs = np.exp(log_probs)
        sum_log_probs = np.sum(exp_log_probs)
        probs = exp_log_probs / sum_log_probs
        return probs

    return log_probs
  
>> x = np.array([10, 2, 10000, 4])
>> print(logsoftmax(x, recover_probs=True))
output: [0., 0., 1., 0.]
```

## Softmax Temperature

In the NLP domain, where the softmax is applied at the output of a classifier to get a probability distribution over tokens. The softmax can be too sure of its predictions and can make other words less likely to pre sampled.    
For example, if we have a statement;

The boy ___ to the market.

with possible answers, $[goes, go, went, comes]$. Assume we get logits of $[38, 20, 40, 39]$ from our classifier to be fed to a softmax function.  
```
>> x = [38, 20, 40, 39]
>> softmax(x)
output: [0.09, 0.00, 0.6, 0.24]
```

If we were to sample from this distribution, $60\%$ of the time, our prediction will be "went" but we are also aware that the answer could also be any of "goes" or "comes" depending on context. The initial logits also show close values of the words but the softmax pushes them away.  
A temperature hyperparameter, $\tau$ is added to the softmax to dampen this extremism. The softmax then becomes
\begin{equation}
   sm(x_i) = \dfrac{
                  e^{\frac{x_i - c}{\tau}}
   }{
                       \sum_{j=1}^{d} e^{\frac{x_j -c}{\tau}}
   }
\end{equation}
where $\tau$ is in $(0, \inf]$.
The temperature parameter increases the sensitivity to low probability candidates and has to be tuned for optimal results. Let's examine different cases of $\tau$

case a: $\tau \to 0$ say $\tau = 0.001$
```
>> softmax(x/100)
output: [0., 0., 1., 0.]
```
This creates a more confident prediction and less likely to sample from unlikely candidates.

case b: $\tau \to \inf$ say $\tau = 100$
```
>> softmax(x/100)
output: [0.25869729, 0.21608214, 0.26392332, 0.26129724]
```
This produces a softer probability distribution over the tokens and results in more diversity in sampling.

## Conclusion

The softmax is an interesting function that requires an in-depth look. We introduced the softmax function and how it can be computed. We then looked at the problems with the naive implementation and how it can lead to numerical instability and proposed a solution. Also, we introduced the log-softmax which makes numerical computation and gradient computation easier. Finally, we discussed the temperature constant used with softmax.
