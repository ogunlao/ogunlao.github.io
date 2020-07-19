---
layout: post
title:  "Breaking down the CTC Loss"
categories: blog
tags: [loss]
comments: true
# categories: coding
# tags: linux

---

The Connectionist Temporal Classification is a type of scoring function for the output of neural networks where the input sequence may not align with the output sequence at every timestep. It was first introduced in the paper by [Alex Graves et al](https://www.cs.toronto.edu/~graves/icml_2006.pdf) for labelling unsegmented phoneme sequence. It has been successfully applied in other classification tasks such as speech recognition, keyword spotting, handwriting recognition, video description. It has become an ubiquitous loss for tasks requiring dynamic alignment of input to output. In this article, we will breakdown the inner workings of the CTC loss computation using the forward-backward algorithm.

For an introductory look at CTC, you can read [Sequence Modeling With CTC](https://distill.pub/2017/ctc/) by Awni Hannun.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

Let's look an automatic speech recognition task where we have to predict the words spoken from the audio data.

![audio converted to text](/images/ctc_loss/asr.png)

Looking at the speech segment, how can we align the words to where they are spoken in the speech segment? Even if it is possible to manually do it for this task, it is not feasible for a large corpus of audio data.

With the CTC alignment, we do not require alignment between input and output sequence (in terms of location). CTC tries all possible alignments of the ground truth to the prediction.

### The CTC Model

Let's get concrete with what we have been talking about by designing a task and apply the CTC loss.

![speech model using ctc ctc](/images/ctc_loss/speech_model.png)

In the model above, we convert the raw audio signal into its spectrum or apply melfiterbanks, which is then passed through a convolution neural network (CNN). CNNs enable us extract features, by looking at a window of the data while performing strided convolutions along the feature dimension of the audio.

The features are then passed through a Recurrent Neural Network (RNN) for decoding. At the decoding stage, observe that some outputs are similar to their previous timesteps indicating that predictions overlap. Also, how should we have dealt with silence in the audio?

Instead of decoding characters, we can decode phonemes, subwords, or even words depending on the task. Let's consider the instance of character decoding for this article.

There are problems which arise when decoding characters. Some characters repeat in words (e.g. 'o' in 'door') and how does the model determine the correct sequence of character to be 'door' and not 'dor'. We can solve this problem by explicitly introducing a blank token into our vocabulary to cater for this. Don not forget we also have a separator token to cater for spaces between words.

For instance, "the door" split into ["t", "h", "e", "d", "o", "o", "r"] tokens will be transformed into ["ε", "t", "ε", "h", "ε", "e", "_", "d", "ε", "o", "ε", "o", "ε", "r"] to cater for blanks and separator. With this, we know that we can only have two similar consecutive tokens only if they are separated by a blank token, "ε".

Given an initial sequence of length $M$, we expand the length of new sequence becomes $2*M + 1$

## Getting into details

At the output of the RNN, we get a vector, which has the length of vocabulary, for each time step of RNN computation. The softmax function is applied to it to get a vector of probabilities. The number of output sequences cannot be more than the number of features from the CNN, so the features has to be estimated accordingly (by taking the maximum length of sequence in vocabulary or some other heuristic).

Let's generate our vocabulary as the standard lowercase alphabets, including our special tokens.

$\{"ε":0, "\_":1', "a": 2, "b":3, ... ,"z":28\}$

![softmax layer from ctc](/images/ctc_loss/softmax_layer_from_ctc.png)

Let $T$ denote the length of the input, $S$ denote the length of the target output i.e $S = 2*M + 1$.

Given these vectors of probability distributions, how do we learn the alignments of the probable predictions? We need a structured way to traverse from the first softmax distribution to the last to represent the word.

### Setting up constrains on the alignment

We eliminate all rows that do not include tokens from the target sequence and then rearrange the tokens to form the output sequence. We copy the required output for the target into a secondary reduced structure. So we only decode on the reduced structure assuring us that only appropriate tokens will be used for decoding.

Also, if a token occur multiple times, we repeat the row in the appropriate location. This becomes our probability matrix, $y_{(s, t)}$

![reduced softmax layer ctc](/images/ctc_loss/reduced_softmax_layer_extract_ctc.png)

### Composing the graph

Now that we have our full grid, we can begin traversing the grid from top-left to bottom right in such a way that; a) the first character in the decoding must be a blank token, 'ε' or the token 'd' b) the last token is either a blank token or 'r' c) the rest of the sequence follows a sequence that monotonically travels down from the top-left to bottom-right.

![probability matrix ctc](/images/ctc_loss/probability_matrix_ctc.png)

To guarantee that the sequence is an expansion of the target sequence, we can only traverse the grid only through these paths. I have attempted to trave all paths in the grid, and you can do it as an exercise too. Two valid paths where both collapse into "door" are shown below;

![](/images/ctc_loss/valid_paths_prob1.png)

![](/images/ctc_loss/valid_paths_prob2.png)

- The sequence can start with a blank token or the first character token and end with a blank token or the last character token. So we have to consider both paths.
- Skips are permitted across a blank token only if the tokens on either side of the blank token are different (because a blank is required to distinguish repetition of a token but not required between distinct tokens)

### Scoring the paths

The score of a path is the product of probabilities of all nodes along the path. For the two paths considered in the examples above.

$score(pathA) = y_{(0,0)}\*y_{(0,1)}\*y_{(0,2)}\*y_{(1,3)}\*y_{(1,4)}\*y_{(2,5)}\*y_{(3,6)}\*y_{(4,7)}\*y_{(5,8)}\*y_{(7,9)}$

$score(pathB) = y_{(1,0)}\*y_{(1,1)}\*y_{(2,2)}\*y_{(3,3)}\*y_{(3,4)}\*y_{(4,5)}\*y_{(5,6)}\*y_{(7,7)}\*y_{(7,8)}\*y_{(8,9)}$

There are an exponential number of such valid paths as can be seen from the graph. The complexity is of the order $\mathcal{O}(|V|^T M)$ where $|V|$ is the length of vocabulary, $T$ is the length of the input and $M$, the number of labels.

Can we find a dynamic programming algorithm for solving this problem? Well, the [viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) is made for this type of problem.
The viterbi algorithm finds the best path to a node by extending the best path to one of its parent nodes. Any other path would necessarily have a lower probability.

Spoiler: But, as good as this sounds, the viterbi algorithm has its drawbacks. It commits to a path or initial alignment early (without exploration) which can lead to suboptimal results.

So, we need to find a way to disallow commit to any valid alignment during training.

## Forward-Backward Algorithm

Instead of only selecting the most likely alignment, we find the expectation over all possible alignments. This allows us to also exploit the existence of subpaths in the graph.

To compute this effectively, we need a forward probability $\alpha_{s, t}$ and backward probability $beta_{s, t}$ where t is the time-step and s is the index of the token considered.

### Forward Algorithm for computing $\alpha_{s, t}$

First, let's create a matrix of zeros of same shape as our probability matrix, $y_{(s, t)}$ to store our $\alpha$ values.

Initialize:

$\alpha$-mat = numpy.zeros_like(y-mat)

$\alpha_{(0, 0)} = y_{(0, 0)}$, $\alpha_{(1, 0)} = y_{(1, 0)}$

$\alpha_{(s, 0)} = 0$ for $s > 1$
  
Iterate forward:

- for t = 1 to T-1:
  - for s = 0 to S:
    - $\alpha_{(s, t)} = (\alpha_{(s, t-1)} + \alpha_{(s-1, t-1)})y_{(s, t)}$ 
if $seq(s) = "ε"$ or seq(s) = seq(s-2)
    - $\alpha_{(s, t)} = (\alpha_{(s, t-1)} + \alpha_{(s-1, t-1)} + \alpha_{(s-2, t-2)})y_{(s, t)}$ otherwise

seq(s) - token at index s e.g. seq(s=1)="d"

![computations of alpha probabilities](/images/ctc_loss/alpha_prob.png)

### Backward algorithm for computing $\beta_{s, t}$

Let's also create a matrix of zeros of same shape as our probability matrix, $y_{(s, t)}$ to store our $\beta$ values.

Initialize:

- $\beta_{(T-1, S-1)} = 1$, $\beta_{(T-1, S-2)} = 1$,
- $\beta_{(T-1, s)} = 0$ for $s < S-2$

Iterate backward:

- for t = T-2 to 0:
  - for s = S-1 to 0:
    - $\beta_{(s, t)} = \beta_{(s, t+1)}y_{(s, t)} + \beta_{(s+1, t+1)}y_{(s+1, t)}$
      if $seq(s) = "ε"$ or $seq(s) = seq(s+2)$
    - $\beta_{(s, t)} = \beta_{(s, t+1)}y_{(s, t)} + \beta_{(s+1, t+1)}y_{(s+1, t)} + \beta_{(s+2, t+2)})y_{(s+2, t)}$ otherwise

![computations of beta probabilities](/images/ctc_loss/beta_prob.png)

### A note of computation

From the computations, observe that we are constantly multiplying probabilities with values less than 0. This can lead to underflow especially for longer sequences. We can improve this computations by performing the computations in the logarithm domain. Products become sums, divisions. For instance;

$\alpha_{s, t} = (\alpha_{s, t-1} + \alpha_{(s-1, t-1)})y_{s, t}$

becomes

$log \alpha_{s,t} = log( e^{log\alpha_{s,t-1}} + e^{log\alpha_{s-1,t-1}}) + log P_{s,t} $

### CTC Loss calculation for each timestep

Now that we have the $\alpha$ and $\beta$ probabilities(or log probabilities), we will compute the joint probability of the sequence at every timestep. This we will call $\gamma_{s,t}$.

$\gamma_{s,t} = \alpha_{s,t}\beta_{s,t}$

Afterwards, we compute the posterior probabilities of the sequence at every timestep by summing along columns. This is the total probability of all paths going through a token $seq(s)$ at timestep t.

$P_{(seq_t), t)} = \sum\limits_{s=0}^{S}\dfrac{\alpha_{s,t}\beta_{s,t}}{y_{s,t}}$

![computations of gamma probabilities](/images/ctc_loss/gamma_prob.png)

Total loss of the model is then;

$\mathcal{l} = -\sum\limits_{t=0}^{T-1}\sum\limits_{s=0}^{S-1}\gamma_{s, t}log~y_{(s, t)}$

Derivatives can then be calculated for back propagation using Autograd. Modern deep leaning libraries such as Pytorch, and TensorFlow have this feature.

## Conclusion

In this article, we explained the connectionist temporal classification loss and how it can be applied in many-to-many input/output classification tasks without alignments. Then, we showed the computations for the forward and backward algorithm used for the decoding. However, we did not mention how the gradients are computed or decoding is done is done during inference. The reader can explore references below for more details.

## References

1. Hannun, "Sequence Modeling with CTC", Distill, 2017.
1. Bhiksha Raj, Connectionist Temporal Classification (CTC) lecture slide, [link](https://deeplearning.cs.cmu.edu/S20/document/slides/lec14.recurrent.pdf), retrieved July 2020.
