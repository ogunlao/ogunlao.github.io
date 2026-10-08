---
title:  "You Don't Really Know Softmax"
#categories: blog
tags: [softmax, numerical_stability]
comments: true
bibliography: 2020-04-26-softmax.bib
---

Softmax function is one of the major functions used in classification models. It is usually introduced early in a machine learning class. It takes as input a real-valued vector of length, d and normalizes it into a probability distribution. It is easy to understand and interpret but at its core are some gotchas that one needs to be aware of. This includes its implementation in practice, numerical stability and applications. The article is an exposé on the topic.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

Softmax is a non-linear function, used majorly at the output of classifiers for multi-class classification. Given a vector $$[x_1, x_2, x_3, \dots x_d]^T$$ for $$i = 1,2, \dots, d$$, the softmax function has the form

$$
\operatorname{sm}(x_i) = \dfrac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}
$$

where d is the number of classes.  
The sum of all the exponentiated values, $$\sum_{j=1}^{d} e^{x_j}$$ is a normalizing constant which helps to ensure that it maintains the properties of a probability distribution i.e., (a) the values must sum to 1 (b) they must be between 0 and 1.

![Softmax classifier](/images/softmax.png "source: ljvmiranda921.github.io")

For example, given a vector $$x = [10, 2, 40, 4]$$, to calculate the softmax of each element;

- exponentiate each value in the vector $$e^x = [e^{10}, e^2, e^{40}, e^4]$$,  
- calculate the sum $$\sum{e^x} = e^{10} + e^2 + e^{40} + e^4 = 2.353\ldots \times 10^{17}$$
- then, divide each $$e^{x_i}$$ by the sum to give

  $$
  \begin{aligned}
  \operatorname{sm}(x) = [\,&9.35762297 \times 10^{-14},\ 3.13913279 \times 10^{-17}, \\
  &1.00000000 \times 10^{0},\ 2.31952283 \times 10^{-16}\,]
  \end{aligned}
  $$


This can be easily implemented in a numerical library like NumPy,

```python
import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    sm_x = exp_x / sum_exp_x
    return sm_x


x = np.array([10, 2, 40, 4])
print(softmax(x))
# [9.35762297e-14 3.13913279e-17 1.00000000e+00 2.31952283e-16]
```

- Questions
  - What do you observe about the output?
  - Will the output sum to 1?

These are pointers to what we will be discussing in the next sections.

## Numerical Stability of Softmax

From the softmax probabilities above, we can deduce that softmax can become numerically unstable for values with a very large range. Consider changing the 3rd value in the input vector to $$10000$$ and re-evaluate the softmax.

```python
x = np.array([10, 2, 10000, 4])
print(softmax(x))
# RuntimeWarning: overflow encountered in exp
# RuntimeWarning: invalid value encountered in divide
# [ 0.  0. nan  0.]
```
'nan' stands for not-a-number. Here, it comes from an overflow: $$e^{10000}$$ becomes infinity, and infinity divided by infinity is undefined. But, why the $$0$$s and $$\text{nan}$$? Are we implying we cannot get a probability distribution from the vector? 
- Question: Can you find out what caused the overflow?

Exponentiating a large number like $$10000$$ leads to a very, very large number, about $$10^{4343}$$. The largest 64-bit float is about $$1.8 \times 10^{308}$$, so $$e^x$$ already overflows for $$x > 709.78$$.

- Can we do better? Well, we can.
Taking our original equation,

$$
\operatorname{sm}(x_i) = \dfrac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}
$$

Let's subtract a constant $$c$$ from the $$x_i$$s

$$
\operatorname{sm}(x_i) = \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}}
$$

We just shift the $$x_i$$ by a constant. If this shifting constant, $$c$$ is the maximum of the vector, $$\max(x)$$, then we can stabilize our softmax computation.

- Question: Do we get the same answer as the original softmax?    
This can be shown to be equivalent to the original softmax function:  
Consider

$$
\begin{aligned}
\operatorname{sm}(x_i) &= \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}} \\
     &= \dfrac{e^{x_i}e^{-c}}{\sum_{j=1}^{d} e^{x_j}e^{-c}} \\
     &= \dfrac{e^{x_i}e^{-c}}{e^{-c}\sum_{j=1}^{d} e^{x_j}}
\end{aligned}
$$

which produces the same initial softmax

$$
\operatorname{sm}(x_i) = \dfrac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}
$$

A NumPy implementation of this stable softmax will look like this:

```python
def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    sm_x = exp_x / sum_exp_x
    return sm_x
```

If we apply it to our old problem:

```python
x = np.array([10, 2, 10000, 4])
print(softmax(x))
# [0. 0. 1. 0.]
```

Great, problem solved !!!

- Question: Why are all other values in the softmax 0. Does it mean they have no probability of occurring?

## Log Softmax

A critical evaluation of the softmax computation shows a pattern of exponentiations and divisions. Can we reduce these computations? We can instead optimize the log softmax. This gives us nice characteristics such as;

1. numerical stability.
1. gradient of log softmax becomes additive since $$\log(a/b) = \log(a) - \log(b)$$
1. lesser computations of divisions and multiplications as addition is less computationally expensive.
1. log is also a monotonically increasing function. We get this property for free

To quote a [stackoverflow answer](https://datascience.stackexchange.com/a/40719)<d-cite key="kevins2018logsoftmax"></d-cite> on using log softmax over softmax:
> There are a number of advantages of using log softmax over softmax including practical reasons like improved numerical performance and gradient optimization. These advantages can be extremely important for implementation especially when training a model can be computationally challenging and expensive. At the heart of using log-softmax over softmax is the use of log probabilities over probabilities, which has nice information theoretic interpretations.
When used for classifiers the log-softmax has the effect of heavily penalizing the model when it fails to predict a correct class. Whether or not that penalization works well for solving your problem is open to your testing, so both log-softmax and softmax are worth using.

If we naively apply the logarithm function to the probability distribution, we get:

```python
x = np.array([10, 2, 10000, 4])
print(softmax(x))
# [0. 0. 1. 0.]

print(np.log(softmax(x)))
# RuntimeWarning: divide by zero encountered in log
# [-inf -inf   0. -inf]
```

We are back to numerical instability, in particular, numerical underflow.

- Question: Why is this so?

The answer lies in taking the logarithm of individual elements. The $$\log(0)$$ is undefined. Can we do better? oh yes!

## Log-Softmax Derivation

$$
\begin{aligned}
\operatorname{sm}(x_i) &= \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}} \\
\log \operatorname{sm}(x_i) &= \log \dfrac{e^{x_i - c}}{\sum_{j=1}^{d} e^{x_j -c}} \\
     &= x_i - c - \log {\sum_{j=1}^{d} e^{x_j -c}}
\end{aligned}
$$

- What if we want to get back our original probabilities?
Well, we can exponentiate and normalize the log softmax or log probability values.

$$
\operatorname{sm}(x_i) = \dfrac{e^{\log \operatorname{sm}(x_i)}}{\sum_{j=1}^{d} e^{\log \operatorname{sm}(x_j)}}
$$

Let's make this concrete via code.

```python
def logsoftmax(x, recover_probs=True):
    # LogSoftMax implementation
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


x = np.array([10, 2, 10000, 4])
print(logsoftmax(x, recover_probs=True))
# [0. 0. 1. 0.]
```

## Softmax Temperature

In the NLP domain, where the softmax is applied at the output of a classifier to get a probability distribution over tokens. The softmax can be too sure of its predictions and can make other words less likely to be sampled.
For example, if we have a statement;

The boy ___ to the market.

with possible answers, $$[\text{goes}, \text{go}, \text{went}, \text{comes}]$$. Assume we get logits of $$[38, 20, 40, 39]$$ from our classifier to be fed to a softmax function.

```python
x = np.array([38, 20, 40, 39])
print(softmax(x).round(2))
# [0.09 0.   0.67 0.24]
```

If we were to sample from this distribution, $$67\%$$ of the time, our prediction will be "went" but we are also aware that the answer could also be any of "goes" or "comes" depending on context. The initial logits also show close values of the words but the softmax pushes them away.  
A temperature hyperparameter, $$\tau$$ is added to the softmax to dampen this extremism. The softmax then becomes

$$
\operatorname{sm}(x_i) = \dfrac{
e^{\frac{x_i - c}{\tau}}
}{
\sum_{j=1}^{d} e^{\frac{x_j -c}{\tau}}
}
$$

where $$\tau$$ is in $$(0, \infty)$$.
The temperature parameter increases the sensitivity to low probability candidates and has to be tuned for optimal results. Let's examine different cases of $$\tau$$

case a: $$\tau \to 0$$ say $$\tau = 0.001$$
```python
print(softmax(x / 0.001).round(2))
# [0. 0. 1. 0.]
```
This creates a more confident prediction and less likely to sample from unlikely candidates.

case b: $$\tau \to \infty$$ say $$\tau = 100$$
```python
print(softmax(x / 100))
# [0.25869729 0.21608214 0.26392332 0.26129724]
```
This produces a softer probability distribution over the tokens and results in more diversity in sampling.

## Conclusion

The softmax is an interesting function that requires an in-depth look. We introduced the softmax function and how it can be computed. We then looked at the problems with the naive implementation and how it can lead to numerical instability and proposed a solution. Also, we introduced the log-softmax which makes numerical computation and gradient computation easier. Finally, we discussed the temperature constant used with softmax.
