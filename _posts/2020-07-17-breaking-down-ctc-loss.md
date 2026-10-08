---
title:  "Breaking down the CTC Loss"
description: How the forward-backward algorithm computes the Connectionist Temporal Classification loss, step by step.
categories: blog
tags: [loss]
comments: true
bibliography: 2020-07-17-ctc-loss.bib
---

The Connectionist Temporal Classification is a type of scoring function for the output of neural networks where the input sequence may not align with the output sequence at every timestep. It was first introduced in the paper by Graves et al.<d-cite key="graves2006ctc"></d-cite> for labelling unsegmented phoneme sequence. It has been successfully applied in other classification tasks such as speech recognition, keyword spotting, handwriting recognition, video description. These tasks require alignment between the input and output which may not be given. Therefore, it has become a ubiquitous loss for tasks requiring dynamic alignment of input to output. In this article, we will break down the inner workings of the CTC loss computation using the forward-backward algorithm<d-cite key="graves2006ctc,raj2020ctc"></d-cite>.

We will not be discussing the decoding methods used during inference such as beam search with CTC or prefix search. For an introductory look at CTC, you can read [Sequence Modeling With CTC](https://distill.pub/2017/ctc/) by Awni Hannun<d-cite key="hannun2017ctc"></d-cite>.

Here's what we will cover:

1. TOC
{:toc}

## Introduction

Let's look at an automatic speech recognition task where we have to predict the words spoken from the audio data.

![audio converted to text](/images/ctc_loss/asr.png)

Looking at the speech segment, how can we align the words to where they are spoken in the speech segment? Even if it is possible to manually do it for this task, it is not feasible for a large corpus of audio data.

With the CTC alignment, we do not require alignment between input and output sequence (in terms of location). CTC tries all possible alignments of the ground truth to the prediction.

### The CTC Model

Let's get concrete with what we have been talking about by designing a task and applying the CTC loss.

![speech model using ctc ctc](/images/ctc_loss/speech_model.png)

In the model above, we convert the raw audio signal into its spectrum or apply mel filterbanks (this is optional and can be performed by a CNN layer), which is then passed through a convolutional neural network (CNN). CNNs enable us to extract features, by looking at a window of the data while performing strided convolutions along the feature dimension of the audio.

The features are then passed through a Recurrent Neural Network (RNN) for decoding. At the decoding stage, if we perform a max decoding at each timestep, we will get tokens of a much longer length than the input, which naturally implies that redundant tokens will be decoded to fill up some of the timesteps. How do we contract the decoded output to represent our predictions? How should we deal with silences in the audio? How should we indicate repetitions of tokens as in "d-oo-r"?

Well, instead of decoding characters, we can decode phonemes, subwords, or even words depending on the task. Let's consider the instance of character decoding for this article.

We can solve these problems by explicitly introducing a blank token into our vocabulary to cater for these dynamics. We further include a separator token to indicate spaces between each word.

Thus, "a door" split into ["ε", "a", "_", "d", "o", "o", "r"] tokens is then transformed into ["ε", "a", "ε", "\_", "ε", "d", "ε", "o", "ε", "o", "ε", "r", "ε"] where the blank token is included. With this, we know that we can have repeating tokens only if they are separated by a blank token, "ε", e.g., "d", "o", "ε", "o", "ε", "r" is allowed and not "d", "o", "o", "ε", "r". The latter contracts into "dor".

In general, given an initial sequence of length $$M$$, the length of the expanded sequence is $$2M + 1$$

## Getting into CTC details

At the output of the RNN, we get a vector, which has the length of vocabulary, for each time step of RNN computation. The softmax function is applied to it to get a vector of probabilities. The number of output labels cannot be more than the number of features from the CNN, so the features have to be estimated accordingly (by taking the maximum length of sequence in vocabulary or some other heuristic).

We will consider a smaller label "door" which should be enough to explain the entire concept succinctly. Let's generate our vocabulary as the standard lowercase alphabets, including our special tokens.

["ε":0, "_":1, "a": 2, "b":3, ... ,"z":28]

![softmax layer from ctc](/images/ctc_loss/softmax_layer_from_ctc.png)

We denote the total number of timesteps by $$T$$, length of the expanded target output by $$S$$, and length of label by $$M$$. So, $$S = 2M + 1$$, e.g., for "door", $$S = 2 \cdot 4 + 1$$

Given these vectors of probability distributions, how do we learn the alignments of the probable predictions? We need a structured way to traverse from the first softmax distribution to the last to represent the word.

### Setting up constraints on the alignment

In principle, we exclude all rows that do not include tokens from the target sequence and then rearrange the tokens to form the output sequence. This is done during training only. At inference, a beam search can be performed on the distribution. So, we copy the required output for the target into a secondary reduced structure and decode on the reduced structure assuring us that only appropriate tokens will be selected for computing loss and gradients.

If a token occurs multiple times in the label, we repeat the rows for similar tokens in their appropriate location. This becomes our probability matrix, $$y_{(s, t)}$$

![reduced softmax layer ctc](/images/ctc_loss/reduced_softmax_layer_extract_ctc.png)

### Composing the graph

Now that we have our full grid, we can begin traversing the grid from top-left to bottom right in such a way that; (a) the first character in the decoding must be a blank token, 'ε' or the first sequence token 'd' (b) the last token is either a blank token or the last sequence token 'r' (c) the rest of the sequence follows a sequence path that monotonically travels down from the top-left to bottom-right.

![probability matrix ctc](/images/ctc_loss/probability_matrix_ctc.png)

To guarantee that the sequence is an expansion of the target sequence, we can only traverse the grid through these valid paths from top-left to bottom-right. I have attempted to trace all paths in the grid, and you can do it as an exercise too. Two valid paths where both collapse into "door" are shown below;

![ctc valid path 1](/images/ctc_loss/valid_paths_prob1.png)

![ctc valid path 2](/images/ctc_loss/valid_paths_prob2.png)

It is easy to trace these paths if we consider the following traversal rules;

- The sequence can start with a blank token or the first character token and end with a blank token or the last character token. So we have to consider both paths.
- Skips are permitted across a blank token **only if the tokens on either side of the blank token are different** because a blank is required to distinguish repetition of a token but not required between distinct tokens

### Scoring the paths

The score of a path is the product of probabilities of all nodes along the path. For the two paths considered in the examples above.

$$
\begin{aligned}
\operatorname{score}(\text{path A}) &= y_{(0,0)} \cdot y_{(0,1)} \cdot y_{(0,2)} \cdot y_{(1,3)} \cdot y_{(1,4)} \cdot y_{(2,5)} \cdot y_{(3,6)} \cdot y_{(4,7)} \cdot y_{(5,8)} \cdot y_{(7,9)} \\
\operatorname{score}(\text{path B}) &= y_{(1,0)} \cdot y_{(1,1)} \cdot y_{(2,2)} \cdot y_{(3,3)} \cdot y_{(3,4)} \cdot y_{(4,5)} \cdot y_{(5,6)} \cdot y_{(7,7)} \cdot y_{(7,8)} \cdot y_{(8,9)}
\end{aligned}
$$

We are required to trace out all the possible paths that contract into "door" and there are an exponential number of such valid paths as can be seen from the graph. The complexity is of the order $$\mathcal{O}(\lvert V \rvert^T)$$ where $$\lvert V \rvert$$ is the length of vocabulary.

Can we find a dynamic programming algorithm for solving this problem? Well, the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) can generate the most likely path, and does not guarantee we get the most likely sequence of labels. It finds the best path to a node by extending the best path to one of its parent nodes. Any other path would necessarily have a lower probability. But, the Viterbi algorithm commits to a path or initial alignment early (without exploration) which can lead to suboptimal results.

## Forward-Backward Algorithm

Instead of only selecting the most likely alignment, we find the expectation over all possible alignments during training. This allows us to also exploit the existence of subpaths in the graph.

To compute this effectively, we need a forward variable $$\alpha_{(s, t)}$$ and backward variable $$\beta_{(s, t)}$$ where $$s$$ is the index of the token considered. The forward variable computes the total probability of emitting the first part of the sequence, $$\operatorname{seq}[0:s]$$, by timestep $$t$$ and being at token $$\operatorname{seq}(s)$$ at that timestep. The backward variable calculates the total probability of emitting the rest of the sequence, up to the last token $$\operatorname{seq}(S-1)$$, after timestep $$t$$, given that we are at token $$\operatorname{seq}(s)$$ at timestep $$t$$. Indices start at 0, and any term with an out-of-range index is taken to be 0.

### Forward Algorithm for computing $$\alpha_{(s, t)}$$

First, let's create a matrix of zeros of same shape as our probability matrix, $$y_{(s, t)}$$ to store our $$\alpha$$ values. The forward algorithm is given by;

Initialize:

- $$\alpha \in \mathbb{R}^{S \times T}$$, a matrix of zeros with the same shape as $$y$$
- $$\alpha_{(0, 0)} = y_{(0, 0)}$$, $$\alpha_{(1, 0)} = y_{(1, 0)}$$
- $$\alpha_{(s, 0)} = 0$$ for $$s > 1$$

Iterate forward:

- for $$t = 1$$ to $$T-1$$:
  - for $$s = 0$$ to $$S-1$$:
    - $$\alpha_{(s, t)} = (\alpha_{(s, t-1)} + \alpha_{(s-1, t-1)})y_{(s, t)}$$
      if $$\operatorname{seq}(s) = \text{“ε”}$$ or $$\operatorname{seq}(s) = \operatorname{seq}(s-2)$$
    - $$\alpha_{(s, t)} = (\alpha_{(s, t-1)} + \alpha_{(s-1, t-1)} + \alpha_{(s-2, t-1)})y_{(s, t)}$$ otherwise

Note that $$\alpha_{(s, t)} = 0$$ for all $$s > 2t + 1$$, the zero boxes in the bottom-left of the figure: these tokens cannot be reached within the first $$t + 1$$ timesteps. The states in the top-right, with $$s < S-2(T-t)$$, still get non-zero values, but there are not enough time-steps left to complete the sequence from them, so they never contribute to the loss.

$$\operatorname{seq}(s)$$ - token at index $$s$$, e.g., $$\operatorname{seq}(s=1)=\text{“d”}$$

<figure class="ctc-figure"><img src="/images/ctc_loss/alpha_prob.svg" alt="computations of alpha probabilities"></figure>

### Backward algorithm for computing $$\beta_{(s, t)}$$

Let's also create a matrix of zeros of same shape as our probability matrix, $$y_{(s, t)}$$ to store our $$\beta$$ values.

Initialize:

- $$\beta \in \mathbb{R}^{S \times T}$$, a matrix of zeros with the same shape as $$y$$
- $$\beta_{(S-1, T-1)} = 1$$, $$\beta_{(S-2, T-1)} = 1$$
- $$\beta_{(s, T-1)} = 0$$ for $$s < S-2$$

Iterate backward:

- for $$t = T-2$$ to $$0$$:
  - for $$s = S-1$$ to $$0$$:
    - $$\beta_{(s, t)} = \beta_{(s, t+1)}y_{(s, t+1)} + \beta_{(s+1, t+1)}y_{(s+1, t+1)}$$
      if $$\operatorname{seq}(s) = \text{“ε”}$$ or $$\operatorname{seq}(s) = \operatorname{seq}(s+2)$$
    - $$\beta_{(s, t)} = \beta_{(s, t+1)}y_{(s, t+1)} + \beta_{(s+1, t+1)}y_{(s+1, t+1)} + \beta_{(s+2, t+1)}y_{(s+2, t+1)}$$ otherwise

Unlike $$\alpha_{(s, t)}$$, $$\beta_{(s, t)}$$ does not include $$y_{(s, t)}$$, the probability of the token at timestep $$t$$ itself; it only covers the timesteps after $$t$$.

Similarly, $$\beta_{(s, t)} = 0$$ for all $$s < S-2(T-t)$$, the zero boxes in the top-right of the figure: the rest of the sequence cannot be completed from these states. The states in the bottom-left, with $$s > 2t + 1$$, get non-zero values but cannot be reached from the start, so they never contribute either. This is why $$\gamma$$ below is zero in both corners.

<figure class="ctc-figure"><img src="/images/ctc_loss/beta_prob.svg" alt="computations of beta probabilities"></figure>

### Computing the probabilities efficiently

From the computations, observe that we are constantly multiplying values less than 1. This can lead to underflow especially for longer sequences. We can improve these computations by performing the computations in the logarithm space. Products become sums, divisions become subtraction. For instance;

$$
\alpha_{s, t} = (\alpha_{s, t-1} + \alpha_{(s-1, t-1)})y_{s, t}
$$

becomes

$$
\log \alpha_{s,t} = \log\left( e^{\log\alpha_{s,t-1}} + e^{\log\alpha_{s-1,t-1}}\right) + \log y_{s,t}
$$

In practice, the $$\log(e^a + e^b)$$ term is computed with the log-sum-exp trick, $$\max(a, b) + \log\left(1 + e^{-\lvert a - b \rvert}\right)$$, so that it never underflows.

### CTC Loss calculation

Now that we have the $$\alpha$$ and $$\beta$$ probabilities(or log probabilities), we will compute the joint probability of the sequence at every timestep. This we will call $$\gamma_{s,t}$$.

$$
\gamma_{s,t} = \alpha_{s,t}\beta_{s,t}
$$

Since $$\alpha_{s,t}$$ covers a path up to and including timestep $$t$$ and $$\beta_{s,t}$$ covers the rest of it, $$\gamma_{s,t}$$ is the total probability of all valid paths that pass through token $$\operatorname{seq}(s)$$ at timestep $$t$$. Summing along a column gives the total probability of the target sequence:

$$
P(\operatorname{seq} \mid x) = \sum\limits_{s=0}^{S-1}\alpha_{s,t}\beta_{s,t}
$$

Every valid path passes through exactly one state at each timestep, so this sum is the same for every $$t$$, which is a useful check on an implementation.

<figure class="ctc-figure"><img src="/images/ctc_loss/gamma_prob.svg" alt="computations of gamma probabilities"></figure>

The CTC loss is the negative log-probability of the target sequence:

$$
\mathcal{L} = -\log P(\operatorname{seq} \mid x) = -\log\left(\alpha_{(S-1, T-1)} + \alpha_{(S-2, T-1)}\right)
$$

The second form follows from the last column, where only the final token and the final blank can end a valid path.

Derivatives can then be calculated for back propagation using Autograd. Modern deep learning libraries such as PyTorch, and TensorFlow have this feature.

### Note

1. The CTC loss algorithm can be applied to both convolutional and recurrent networks. For recurrent networks, it is possible to compute the loss at each timestep in the path or make use of the final loss, depending on the use case.
2. Graves et al.<d-cite key="graves2006ctc"></d-cite> define the backward variable to include $$y_{(s, t)}$$, i.e., their $$\beta$$ equals $$\beta_{(s, t)}\,y_{(s, t)}$$ here. With that convention, the column sum becomes $$\sum_s \alpha_{s,t}\beta_{s,t} / y_{s,t}$$. Both give the same $$P(\operatorname{seq} \mid x)$$. This article follows the convention of the CMU lecture slides<d-cite key="raj2020ctc"></d-cite>.
3. The loss value itself only needs the forward variables, $$\alpha_{s, t}$$, as shown above; this matches PyTorch's `ctc_loss`. The backward variables are what make the gradients cheap to compute, as proposed in the seminal paper by Graves et al.<d-cite key="graves2006ctc"></d-cite>. The CMU lecture<d-cite key="raj2020ctc"></d-cite> instead trains with the expected divergence, $$-\sum_t \sum_s \frac{\gamma_{s,t}}{P(\operatorname{seq} \mid x)} \log y_{s,t}$$, which has a different value but the same gradient with respect to $$y$$ as $$\mathcal{L}$$.

**Update**
Oct 8, 26: Fixed errors in the forward-backward algorithm from an earlier version of this article: the forward and backward recursions and the loss formula. The notes on zero-valued states were also restated in 0-based indexing to match the figures.

## Conclusion

In this article, we explained the connectionist temporal classification loss and how it can be applied in many-to-many input/output classification tasks without alignments. Then, we showed the computations for the forward and backward algorithm used for training the model.
