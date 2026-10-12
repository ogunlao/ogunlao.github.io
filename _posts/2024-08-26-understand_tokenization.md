---
title:  "Build a BPE Tokenizer from Scratch, and See Where It Fails"
description: Build a byte pair encoding (BPE) tokenizer from scratch in Python, step by step, from the original subword algorithm to a GPT-4-style byte-level tokenizer, and see where tokenization fails.
#categories: blog
tags: [tokenization, bpe]
comments: true
bibliography: 2024-08-26-tokenization.bib
---

Tokenization is how text is split into small units, called tokens, and mapped to the integers that a large language model (LLM) actually sees. Every prompt you send to an LLM goes through a tokenizer first, and many strange behaviours of LLMs can be traced back to it. In this article, I will break down the steps to implement a byte pair encoding (BPE) tokenizer from scratch in Python, starting from the algorithm in the original paper by Sennrich et al.<d-cite key="sennrich2016bpe"></d-cite>. We will then extend it to UTF-8 bytes, and implement the pre-tokenization strategy used by LLMs like GPT-4. Let's go!

All the code blocks are meant to be run in order, in a single Python session or notebook (Python 3.9 or later). The output of each block is shown as comments below it.

Here's what we will cover:
1. TOC
{:toc}

## Introduction

The seminal paper by Sennrich et al.<d-cite key="sennrich2016bpe"></d-cite> introduced byte pair encoding (BPE)<d-cite key="gage1994bpe"></d-cite> to neural machine translation. BPE is itself a data compression algorithm that existed long before it was used for tokenization.

Before BPE, neural machine translation models mostly used word-level vocabularies, with character-level models as the main alternative. Both have their shortcomings. Character-level tokenization produces long sequences, while word-level tokenization does not do well with rare and out-of-vocabulary words. BPE was introduced to cater for rare words using subword tokenization, which provides a balance between the two. The BPE algorithm has gone through some modifications and improvements since then, which we will also discuss and implement.

## What is BPE tokenization?

Consider a sentence in English: "The boy looks good."

- Character-level tokenization splits on each character: `"T", "h", "e", " ", "b", "o", "y", " ", "l", "o", "o", "k", "s", " ", "g", "o", "o", "d", "."` $$\rightarrow$$ 19 tokens
- Word-level tokenization splits on words: `"The", "boy", "looks", "good", "."` $$\rightarrow$$ 5 tokens
- Subword tokenization splits it into subwords: `"The", " boy", " lo", "oks", " good", "."` $$\rightarrow$$ 6 tokens

Looking at the examples above, character-level tokenization can easily cover the entire vocabulary with a small set of symbols, but it requires the most tokens, and a single character likely does not carry any meaning. It also means a model takes more steps to process or generate the same sentence. Conversely, word-level tokenization is more compact and uses fewer tokens, but there is a long tail of words that the tokenizer may never see during training, and modelling new words is complicated. The concept of a word is also relative, as languages like Chinese, Japanese and Thai do not mark word boundaries with spaces, unlike English. Subword tokenization is the middle ground. It can be adapted easily to model new words and vocabularies, it can be learned from the training data using a statistical process, and it scales well to multilingual systems.

The original BPE compression algorithm<d-cite key="gage1994bpe"></d-cite>, as described by Sennrich et al.<d-cite key="sennrich2016bpe"></d-cite>,

> iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.

Sennrich et al. adapt this to word segmentation: instead of bytes, they merge characters, and instead of replacing a pair with an unused byte, they add the merged pair as a new symbol (a character n-gram) to the vocabulary. The merges are learned from the training data, so frequent words end up as single tokens while rare words are split into smaller, known pieces.

## Build a basic BPE tokenizer

Let's start with the BPE algorithm as described by Sennrich et al.<d-cite key="sennrich2016bpe"></d-cite>. We will build it step by step, following the second paragraph in Section 3.2 of the paper, which concretely describes the algorithm.

### Step 1: Initialize the character vocabulary

> [W]e initialize the symbol vocabulary with the character vocabulary, and represent each word as a sequence of characters, plus a special end-of-word symbol '·', which allows us to restore the original tokenization after translation.

I will use "\_" as the end-of-word symbol instead of "·", as it is easier to type. For this tutorial, our character vocabulary is the lowercase English alphabet plus the end-of-word symbol. Each symbol is mapped to an integer ID.

```python
# Character-level base vocabulary with '_' as the end-of-word symbol
char_vocab = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25, "_": 26}
```

If we wanted to cover more characters, we could take the first 256 Unicode characters (ASCII and Latin-1) with `{chr(idx): idx for idx in range(256)}`. We will keep the small vocabulary for now, and come back to this idea when we build a byte-level tokenizer.

### Step 2: Count the words in the corpus

Now, let's represent each word as a sequence of characters separated by spaces, ending with the special "\_" symbol. Each word is paired with its frequency count in the corpus.

```python
import collections

def build_bpe_word_freqs(text: str, end_of_word: str = "_") -> dict[str, int]:
    """
    Counts word frequencies in a text corpus and formats them for BPE training.
    Example:
        'lower' -> 'l o w e r _'
    """
    # Normalize space and split text into words
    words = text.strip().split()

    # Count raw frequencies
    raw_counts = collections.Counter(words)

    # Format each word as a space-separated sequence of characters ending with the marker
    formatted_freqs = {}
    for word, count in raw_counts.items():
        formatted_word = " ".join(list(word)) + f" {end_of_word}"
        formatted_freqs[formatted_word] = count

    return formatted_freqs
```

Let's try it on a toy corpus. It is similar to the example in Algorithm 1 of the paper, with different counts.

```python
text = "low lower newest newest newest widest widest widest widest low low low low"
word_freqs = build_bpe_word_freqs(text, "_")
print("Formatted BPE Word Frequencies:")
for k, v in word_freqs.items():
    print(f"{repr(k)}: {v}")

# Formatted BPE Word Frequencies:
# 'l o w _': 5
# 'l o w e r _': 1
# 'n e w e s t _': 3
# 'w i d e s t _': 4
```

`word_freqs` is our training corpus representation. It contains the words we are learning BPE merges from, along with how often they occur. Notice that we no longer need the original text: the algorithm only works on this dictionary.

### Step 3: Count the symbol pairs

> We iteratively count all symbol pairs and replace each occurrence of the most frequent pair (‘A’, ‘B’) with a new symbol ‘AB’. Each merge operation produces a new symbol which represents a character n-gram.

First, we count how often each pair of adjacent symbols occurs. Each pair inside a word is counted as many times as the word occurs in the corpus.

```python
# Taken from Algorithm 1 in the BPE paper
def get_stats(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

print(dict(get_stats(word_freqs)))
# {('l', 'o'): 6, ('o', 'w'): 6, ('w', '_'): 5, ('w', 'e'): 4, ('e', 'r'): 1, ('r', '_'): 1, ('n', 'e'): 3, ('e', 'w'): 3, ('e', 's'): 7, ('s', 't'): 7, ('t', '_'): 7, ('w', 'i'): 4, ('i', 'd'): 4, ('d', 'e'): 4}
```

For example, ('l', 'o') occurs 6 times: 5 times in "low" and once in "lower".

### Step 4: Merge the most frequent pair

> Frequent character n-grams (or whole words) are eventually merged into a single symbol, thus BPE requires no shortlist.

Next, we replace every occurrence of a pair with the merged symbol. The regular expression makes sure we only merge whole symbols: `(?<!\S)` and `(?!\S)` check that the pair is not part of a longer symbol on either side.

```python
import re

# Also taken from Algorithm 1 in the BPE paper
def merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

print(merge_vocab(('e', 's'), word_freqs))
# {'l o w _': 5, 'l o w e r _': 1, 'n e w es t _': 3, 'w i d es t _': 4}
```

"e" and "s" are now a single symbol "es" in "newest" and "widest".

### Step 5: Repeat

The rest of Section 3.2 tells us when to stop:

> The final symbol vocabulary size is equal to the size of the initial vocabulary, plus the number of merge operations – the latter is the only hyperparameter of the algorithm. For efficiency, we do not consider pairs that cross word boundaries. The algorithm can thus be run on the dictionary extracted from a text, with each word being weighted by its frequency.

So, we repeat steps 3 and 4 for a fixed number of merges, and keep a list of the merges in the order they were learned.

```python
num_merges = 10
vocab = word_freqs.copy()
merges = []

print("Initial Vocabulary State:", vocab)
print("\n--- Iterative BPE Merges ---")
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(best)
    print(f"Merge #{i+1}: {best} -> frequency: {pairs[best]}")

print("\nFinal Vocabulary State:", vocab)
print("Learned Merges:", merges)

# Initial Vocabulary State: {'l o w _': 5, 'l o w e r _': 1, 'n e w e s t _': 3, 'w i d e s t _': 4}
#
# --- Iterative BPE Merges ---
# Merge #1: ('e', 's') -> frequency: 7
# Merge #2: ('es', 't') -> frequency: 7
# Merge #3: ('est', '_') -> frequency: 7
# Merge #4: ('l', 'o') -> frequency: 6
# Merge #5: ('lo', 'w') -> frequency: 6
# Merge #6: ('low', '_') -> frequency: 5
# Merge #7: ('w', 'i') -> frequency: 4
# Merge #8: ('wi', 'd') -> frequency: 4
# Merge #9: ('wid', 'est_') -> frequency: 4
# Merge #10: ('n', 'e') -> frequency: 3
#
# Final Vocabulary State: {'low_': 5, 'low e r _': 1, 'ne w est_': 3, 'widest_': 4}
# Learned Merges: [('e', 's'), ('es', 't'), ('est', '_'), ('l', 'o'), ('lo', 'w'), ('low', '_'), ('w', 'i'), ('wi', 'd'), ('wid', 'est_'), ('n', 'e')]
```

Looking at the output, we can observe several things;
- The learned merges build on each other. "es" is merged first, then "est", then "est\_". The suffix "est\_" is shared by "newest" and "widest", so it is learned before any full word.
- When two pairs have the same frequency, `max` returns the one it saw first. That is why ('e', 's') was merged before ('s', 't') and ('t', '\_'), which also occur 7 times.
- The final vocabulary state shows that the frequent words, "low" and "widest", are now single symbols, "low\_" and "widest\_". The rare word "lower" is still split into "low", "e", "r", "\_", and "newest" into "ne", "w", "est\_". This is exactly the behaviour we want: frequent words become one token, and rare words are built from smaller known pieces.
- Our vocabulary now has $$27 + 10 = 37$$ symbols: the 27 characters and the 10 merged symbols.

### Step 6: Put it together in a tokenizer class

Let's now build the complete tokenizer class that can train on a text, encode text into token IDs, and decode the IDs back to the original text. To encode a new word, we split it into characters and apply the learned merges in the order they were learned. Each merged symbol gets the next available integer ID after the base characters.

```python
class BPETokenizer:
    def __init__(self, char_vocab: dict[str, int], end_of_word: str = "_"):
        # The base character vocabulary
        self.char_vocab = char_vocab.copy()
        self.end_of_word = end_of_word
        # The complete vocabulary (base characters + merged symbols)
        self.vocab = self.char_vocab.copy()
        # List of learned merges as tuples of symbols, e.g. ('e', 's')
        self.merges = []
        self.idx2word = {idx: symbol for symbol, idx in self.vocab.items()}

    def train(self, text: str, num_merges: int):
        training_vocab = build_bpe_word_freqs(text, self.end_of_word)
        self.vocab = self.char_vocab.copy()
        self.merges = []

        # New merged symbols get IDs after the base vocabulary
        next_token_id = max(self.char_vocab.values()) + 1

        for i in range(num_merges):
            pairs = get_stats(training_vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            training_vocab = merge_vocab(best, training_vocab)
            self.merges.append(best)

            # Record the new merged symbol in the vocabulary
            merged_symbol = ''.join(best)
            if merged_symbol not in self.vocab:
                self.vocab[merged_symbol] = next_token_id
                next_token_id += 1

        self.idx2word = {idx: symbol for symbol, idx in self.vocab.items()}

    def segment(self, word: str) -> str:
        """
        Apply the learned BPE merge operations, in order, to segment a word.
        """
        word_str = ' '.join(list(word) + [self.end_of_word])
        for pair in self.merges:
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            word_str = p.sub(''.join(pair), word_str)
        return word_str

    def encode(self, text: str) -> list[int]:
        """
        Encodes text into a list of BPE subword token IDs.
        """
        subwords = []
        for word in text.split():
            subwords.extend(self.segment(word).split())
        return [self.vocab[subword] for subword in subwords]

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of BPE subword token IDs back into text.
        """
        text = "".join(self.idx2word[token] for token in tokens)
        return text.replace(self.end_of_word, " ").strip()
```

Let's train this tokenizer on a short text. I will use the abstract of the BPE paper<d-cite key="sennrich2016bpe"></d-cite> as our corpus, and learn 10 merges.

```python
sample_corpus = """Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character n-gram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English→German and English→Russian by up to 1.1 and 1.3 BLEU, respectively."""

bpe_tok = BPETokenizer(char_vocab=char_vocab)
bpe_tok.train(sample_corpus, num_merges=10)

print("Learned Merges:", bpe_tok.merges)
print("New symbols in the vocabulary:")
for symbol, idx in bpe_tok.vocab.items():
    if idx >= len(char_vocab):
        print(f"  {symbol!r}: {idx}")

# Test segmentation on an unseen word
print("Segmented 'lower':", bpe_tok.segment("lower"))
print("Segmented 'translation':", bpe_tok.segment("translation"))

# Learned Merges: [('a', 'n'), ('e', '_'), ('s', '_'), ('o', 'n'), ('t', 'i'), ('d', '_'), ('o', 'r'), ('ti', 'on'), ('i', 'n'), ('l', 'a')]
# New symbols in the vocabulary:
#   'an': 27
#   'e_': 28
#   's_': 29
#   'on': 30
#   'ti': 31
#   'd_': 32
#   'or': 33
#   'tion': 34
#   'in': 35
#   'la': 36
# Segmented 'lower': l o w e r _
# Segmented 'translation': t r an s la tion _
```

The tokenizer has learned some common n-grams of English, like "an", "in" and "tion", and word endings like "e\_", "s\_" and "d\_". None of the 10 merges matches a pair of adjacent characters in "lower", so it stays as characters. With more merges (real tokenizers use tens of thousands), longer pieces and whole words get merged.

Now that we have learned the vocabulary from text, we can use the tokenizer to encode and decode text.

```python
sentence = "the man has an identity"
encoded = bpe_tok.encode(sentence)
print("Encoded sequence of IDs:", encoded)

# Let's see what each token ID maps to in the vocabulary
print("Corresponding subword pieces:", [bpe_tok.idx2word[idx] for idx in encoded])

# Encoded sequence of IDs: [19, 7, 28, 12, 27, 26, 7, 0, 29, 27, 26, 8, 3, 4, 13, 31, 19, 24, 26]
# Corresponding subword pieces: ['t', 'h', 'e_', 'm', 'an', '_', 'h', 'a', 's_', 'an', '_', 'i', 'd', 'e', 'n', 'ti', 't', 'y', '_']
```

Our 23-character sentence is encoded into 19 tokens. Now, let's decode the token IDs back to text.

```python
print(bpe_tok.decode(encoded))
# the man has an identity
```

The BPE tokenizer as implemented here has some limitations. Observe what happens when we try to encode a character that is not in our base vocabulary.

```python
bpe_tok.encode("The man")
# KeyError: 'T'
```

We could fix that by adding an `<unk>` symbol for unknown characters, but then the tokenizer would lose information. We could also add more characters to the base vocabulary. However, the initial character vocabulary can grow to tens of thousands of characters for multilingual tokenization if we consider the range of characters in Unicode. Therefore, newer implementations, starting with the GPT-2 tokenizer<d-cite key="radford2019gpt2"></d-cite>, build the tokenizer from the byte representation of the text instead.

## Build a byte-level tokenizer

### Code points and bytes

In Python, it is easy to get the Unicode code point of a character using the `ord()` function, and `chr()` does the inverse.

```python
print(ord("o"), ord("à"))
# 111 224
print(chr(111), chr(224))
# o à
```

However, note that some characters that are rendered as a single unit are actually a sequence of code points: a base character followed by one or more combining marks. `ord()` will fail if you directly apply it to such a character.

```python
print(ord("ọ̀"))
# TypeError: ord() expected a character, but string of length 2 found
```

```python
print([ord(x) for x in "ọ̀"])
# [7885, 768]
print([chr(x) for x in [7885, 768]])
# ['ọ', '̀']
```

"ọ̀" (o with a dot below and a grave accent, common in Yorùbá) is made of "ọ" (7885) followed by a combining grave accent (768). There is no single code point for it. Even worse, the same visible character can be written with different sequences of code points, and they are different strings to Python. Unicode normalization, e.g. NFC, which composes characters where possible, maps all of them to a single form.

```python
import unicodedata

spellings = ["\u1ecd\u0300", "\u00f2\u0323", "o\u0323\u0300"]  # ọ + grave, ò + dot below, o + dot below + grave
for s in spellings:
    print(s, [ord(x) for x in s])
print("Distinct strings:", len(set(spellings)))
print("Distinct strings after NFC:", len({unicodedata.normalize("NFC", s) for s in spellings}))

# ọ̀ [7885, 768]
# ọ̀ [242, 803]
# ọ̀ [111, 803, 768]
# Distinct strings: 3
# Distinct strings after NFC: 1
```

This means that you may have to normalize your text before training a tokenizer, and before encoding with it. Otherwise, the same word written in two different ways will be tokenized differently.

Using code points directly as our base vocabulary would also make it too large. Unicode has 1,114,112 possible code points (from U+0000 to U+10FFFF), and nearly 160,000 of them (159,801 as of Unicode 17.0) are assigned to characters. That's why we prefer to use byte-level representations of the text. UTF-8<d-cite key="yergeau2003utf8"></d-cite> encodes every code point as a sequence of 1 to 4 bytes, and a byte can only take 256 values. Here is what it looks like in Python:

```python
text = "àwọn ọmọ Yorùbá (Yoruba people)"
byte_ids = list(text.encode("utf-8"))
print(len(text), "characters ->", len(byte_ids), "bytes")
print(byte_ids)

# 31 characters -> 40 bytes
# [195, 160, 119, 225, 187, 141, 110, 32, 225, 187, 141, 109, 225, 187, 141, 32, 89, 111, 114, 195, 185, 98, 195, 161, 32, 40, 89, 111, 114, 117, 98, 97, 32, 112, 101, 111, 112, 108, 101, 41]
```

Observe that each visually rendered character can take more than one byte, depending on its code point. ASCII characters like "w" and "Y" take one byte (119 and 89), "à" takes two bytes (195, 160), and "ọ" takes three bytes (225, 187, 141). We have discussed this briefly to motivate the tokenizer. For a more in-depth explanation of Unicode and UTF-8, I will refer readers to the article by Nathan Reed<d-cite key="reed2017unicode"></d-cite> or the UTF-8 specification<d-cite key="yergeau2003utf8"></d-cite>.

### Build a basic byte-level tokenizer

This implementation is similar to minbpe<d-cite key="karpathy2024minbpe"></d-cite>. The byte-level tokenizer works like our basic tokenizer, but the base vocabulary is the 256 possible byte values, and merges are pairs of integer IDs. The first merge gets the ID 256, the next 257, and so on, so the final vocabulary size is $$256$$ plus the number of merges. Here, I show the entire class, as we have previously discussed the major tokenizer methods. The main differences are;

- `merges` is a dictionary from a pair of token IDs to the new token ID, e.g. `{(101, 32): 256}`.
- `get_stats` counts pairs over the whole sequence of bytes, so merges can cross word boundaries. We will come back to this.
- `encode` applies the merges by rank: at every step, it merges the pair that was learned earliest during training, until no learned pair is left.
- `decode` concatenates the bytes of each token and decodes them back to a string with UTF-8.
- The tokenizer can start from some default merges. I will explain why later.

```python
class ByteLevelTokenizer:
    def __init__(self, default_merges: dict[tuple[int, int], int] = None):
        self.merges = default_merges.copy() if default_merges else {}
        self.vocab = self.get_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        tokens = self.byte_encode(text)
        self.merges = self.train_tokens(tokens, vocab_size, verbose)
        self.vocab = self.get_vocab()

    def get_vocab(self) -> dict[int, bytes]:
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        # Sort merges by their token IDs to build up dependencies in the correct order
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (tk1, tk2), tk_idx in sorted_merges:
            vocab[tk_idx] = vocab[tk1] + vocab[tk2]
        return vocab

    def byte_encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def merge_tokens(self, tokens: list[int], ids: tuple[int, int], token: int) -> list[int]:
        if len(tokens) < 2:
            return tokens

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == ids[0] and tokens[i+1] == ids[1]:
                new_tokens.append(token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def train_tokens(self, tokens: list[int], vocab_size: int, verbose: bool) -> dict[tuple[int, int], int]:
        merges = self.merges.copy()

        # First apply any pre-existing merges to our training tokens
        sorted_merges = sorted(merges.items(), key=lambda x: x[1])
        for (tk1, tk2), tk_idx in sorted_merges:
            tokens = self.merge_tokens(tokens, (tk1, tk2), tk_idx)

        # Calculate how many new merges we have left to learn
        total_iters = vocab_size - 256 - len(merges)

        for i in range(total_iters):
            stats = self.get_stats(tokens)
            if not stats:
                break
            tk1, tk2 = max(stats, key=stats.get)

            # The new token ID follows the existing merges
            new_tk = 256 + len(merges)
            merges[(tk1, tk2)] = new_tk
            if verbose:
                print(f"Merging tokens {tk1} and {tk2} into token {new_tk}")
            tokens = self.merge_tokens(tokens, (tk1, tk2), new_tk)
        return merges

    def get_stats(self, tokens: list[int]) -> dict[tuple[int, int], int]:
        token_stats = {}
        for tk1, tk2 in zip(tokens, tokens[1:]):
            token_stats[(tk1, tk2)] = token_stats.get((tk1, tk2), 0) + 1

        return token_stats

    def encode(self, text: str) -> list[int]:
        tokens = self.byte_encode(text)
        while len(tokens) > 1:
            stats = self.get_stats(tokens)
            # Pick the pair that was merged earliest during training
            ids = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if ids not in self.merges:
                break
            idx = self.merges[ids]
            tokens = self.merge_tokens(tokens, ids, idx)

        return tokens

    def decode(self, ids: list[int]) -> str:
        decoded = b"".join(self.vocab[idx] for idx in ids)
        text = decoded.decode("utf-8", errors="replace")
        return text
```

Note that a learned token can hold an incomplete UTF-8 sequence, e.g. only the first byte of "è". Decoding the full list of tokens works, but decoding such a token on its own is not valid UTF-8. `errors="replace"` replaces these invalid bytes with the replacement character "�" instead of raising an error.

Let's train this tokenizer on some Yorùbá text, the first paragraph of the [Ìran Yorùbá](https://yo.wikipedia.org/wiki/%C3%8Cran_Yor%C3%B9b%C3%A1) article on the Yorùbá Wikipedia (without its reference markers). We will learn 94 merges, i.e. a vocabulary size of 350, encode a sentence that is not in the training text, and look at the longest tokens the tokenizer has learned.

```python
yoruba_text = """Ìran Yorùbá, àwọn ọmọ Yorùbá tàbí Ọmọ káàárọ̀-oòjíire, jẹ́ ọmọ orílẹ̀-èdè Áfíríkà, ní apá Ìwọ̀-Oòrùn rẹ̀, tí wọ́n ń gbé apá kan ní ilẹ̀ Nàìjíríà, Bẹ̀nẹ̀, àti Togo, wọ́n sì ń pè gbogbo agbègbè yìí ní Ilẹ̀ Yorùbá. Ẹ lè ri púpọ̀ nínú wọ́n ní àwọn ìpínlẹ̀ bíi ìpínlẹ̀ Ẹdó, Ìpínlẹ̀ Èkìtì, ìpínlẹ̀ Èkó, Ìpínlẹ̀ Kwara, ìpínlẹ̀ Kogí, ìpínlẹ̀ Ògùn, Ìpínlẹ̀ Oǹdó, ìpínlẹ̀ Ọ̀ṣun, ìpínlẹ̀ Ọ̀yọ́. Ẹ tún le rí wọ́n ní ìpínlẹ̀ to wa nínú orílẹ̀-èdè Olómìnira Benin (Dahomey), ní orílẹ̀-èdè Sàró (Sierra Leone), àti ní àwọn orílẹ̀-èdè miiran bíi Togo, Brazil, Cuba, Haiti, Amẹ́ríkà ati Venezuela. Àwọn Yorùbá wà l’árá àwọn to tóbí ju ní orílẹ̀-èdè Nàìjíríà. Ó le jẹ́ pe àwọn ló pọ̀ jù, abí kí wọ́n jẹ́ ìkejì, tàbí ẹ̀yà kẹta tó pọ̀ jùlọ ní orílẹ̀-èdè Nàìjíríà."""

byte_tokenizer = ByteLevelTokenizer()
byte_tokenizer.train(yoruba_text, vocab_size=350)

sentence = "E nilo lati dọbalẹ. Ni asa wa ati ni èdè wa, òwó se pataki gidigan"
ids = byte_tokenizer.encode(sentence)
print(len(sentence.encode("utf-8")), "bytes ->", len(ids), "tokens")
print([byte_tokenizer.decode([idx]) for idx in ids])
print(byte_tokenizer.decode(ids))
print(byte_tokenizer.decode(ids) == sentence)

# The longest tokens learned during training
learned = [byte_tokenizer.vocab[idx] for idx in range(256, 350)]
longest = sorted(learned, key=len, reverse=True)[:5]
print([tok.decode("utf-8", errors="replace") for tok in longest])

# 74 bytes -> 54 tokens
# ['E', ' ', 'n', 'i', 'l', 'o', ' l', 'a', 'ti ', 'd', 'ọ', 'b', 'a', 'l', 'ẹ', '. ', 'N', 'i ', 'a', 's', 'a', ' ', 'w', 'a', ' a', 'ti ', 'n', 'i', ' �', '�', 'd', 'è ', 'w', 'a', ', �', '�', 'w', 'ó', ' ', 's', 'e', ' p', 'a', 't', 'a', 'k', 'i ', 'g', 'i', 'd', 'i', 'g', 'a', 'n']
# E nilo lati dọbalẹ. Ni asa wa ati ni èdè wa, òwó se pataki gidigan
# True
# ['ní orílẹ̀-èdè ', 'orílẹ̀-èdè ', 'orílẹ̀-èd', 'orílẹ̀-è', 'Nàìjíríà']
```

The tokenizer decodes the sentence back to the original text, and the 74 bytes are compressed into 54 tokens. Let's look at what it has learned;

- **Characters.** `"ọ"` and `"ẹ"` take 3 bytes each in UTF-8, but they are frequent in Yorùbá, so each of them is now a single token.
- **Subwords.** `"ti "` is a token, because the word "àti" ("and") appears three times in the training text, twice with a tone mark and once without. Since the tone mark is on the "à", the tokenizer learns the ending `"ti "`, which our sentence uses twice, in "lati" and "ati". Similarly, `"i "` comes from words like "bíi" and "ri".
- **Whole words and phrases.** The longest learned tokens are words that are repeated in the training text, like `"Nàìjíríà"`, and even the phrase `"ní orílẹ̀-èdè "` ("in the country"). Notice how the merges build on each other: `"orílẹ̀-è"`, `"orílẹ̀-èd"`, `"orílẹ̀-èdè "`, then `"ní orílẹ̀-èdè "`.
- **Partial characters.** Notice the pieces `" �"`, `"�"` and `", �"`, `"�"`. "è" and "ò" are 2 bytes each, and their first byte was merged with the space (or comma and space) before it. Decoded on their own, these tokens are not valid UTF-8, but decoding the full list gives back "è" and "ò".

Our sentence still has many single-character tokens, because it is written without most of the tone marks used in the training text. For example, "nilo" and "lati" have no tone marks here, while the training text uses words like "ní" and "àti" with tone marks. To the tokenizer, "ni" and "ní" are different byte sequences.

Why not learn more merges? Our training text is only 1,052 bytes long. After about 120 merges, every remaining pair of tokens occurs only once, so `max` simply picks the first pair in the text, and the tokenizer starts memorizing the text from the beginning. With a vocabulary size of 400, the first 53 characters of the article, "Ìran Yorùbá, àwọn ọmọ Yorùbá tàbí Ọmọ káàárọ̀-oòjíire", become a single token! Real tokenizers are trained on gigabytes of text, where this does not happen, and learn tens of thousands of merges, e.g. about 100,000 tokens for GPT-4's `cl100k_base`.

By starting with only 256 byte values, we can tokenize any Unicode text, including emojis and scripts that never appeared in the training data. They will just take more tokens.

```python
emoji_text = "Hello 👋 世界"
print(byte_tokenizer.decode(byte_tokenizer.encode(emoji_text)) == emoji_text)
# True
```

### Start from default merges

We have also extended the byte-level tokenizer to take in some default merges that we want to be part of the vocabulary. This may be the case for text-to-speech or low-resource languages, where we want to bias the tokenizer to have some basic subwords in that language, even though we have a limited amount of training data to learn this tokenization pattern. The default merges are kept, and new merges are learned on top of them.

Let's start with a function to automatically create the merges dictionary from a list of strings.

```python
def create_default_merges(strings: list[str]) -> dict[tuple[int, int], int]:
    """
    Generates a BPE merges dictionary from a list of target strings.
    Breaks down strings into sequential pair merges starting at token ID 256.
    """
    merges = {}
    next_token_id = 256

    for s in strings:
        # Encode the string into its raw UTF-8 bytes
        tokens = list(s.encode("utf-8"))

        # We need at least 2 bytes to perform a merge
        while len(tokens) > 1:
            # Try to see if we can find existing merges to apply first
            # so we reuse previously created sub-merges
            merged_any = False
            for (p1, p2), tid in merges.items():
                # Look for the pair in our current token sequence
                for idx in range(len(tokens) - 1):
                    if tokens[idx] == p1 and tokens[idx+1] == p2:
                        tokens[idx:idx+2] = [tid]
                        merged_any = True
                        break
                if merged_any:
                    break

            if merged_any:
                continue

            # If no existing merges apply, create a new merge for the first pair
            pair = (tokens[0], tokens[1])
            merges[pair] = next_token_id
            tokens[0:2] = [next_token_id]
            next_token_id += 1

    return merges
```

```python
basic_language_tokens = [" at", " how"]
default_merges = create_default_merges(basic_language_tokens)
print("Generated merges:", default_merges)

low_resource_tokenizer = ByteLevelTokenizer(default_merges=default_merges)
print("Default tokens:", [low_resource_tokenizer.decode([idx]) for idx in default_merges.values()])

low_resource_tokenizer.train("This is how we see all these things", vocab_size=270, verbose=True)
ids = low_resource_tokenizer.encode("This is how we see it")
print([low_resource_tokenizer.decode([idx]) for idx in ids])

# Generated merges: {(32, 97): 256, (256, 116): 257, (32, 104): 258, (258, 111): 259, (259, 119): 260}
# Default tokens: [' a', ' at', ' h', ' ho', ' how']
# Merging tokens 104 and 105 into token 261
# Merging tokens 101 and 32 into token 262
# Merging tokens 84 and 261 into token 263
# Merging tokens 263 and 115 into token 264
# Merging tokens 264 and 32 into token 265
# Merging tokens 265 and 105 into token 266
# Merging tokens 266 and 115 into token 267
# Merging tokens 267 and 260 into token 268
# Merging tokens 268 and 32 into token 269
# ['This is how ', 'w', 'e ', 's', 'e', 'e ', 'i', 't']
```

`" at"` and `" how"` were built with 5 merges (`" a"`, `" at"`, `" h"`, `" ho"`, `" how"`), so the tokenizer starts with them in its vocabulary, and the IDs of new merges start from 261. The training then reuses `" how"` in one of its merges.

However, look at the learned tokens. Since we count pairs over the whole sequence, the tokenizer merged across word boundaries and learned `"This is how "` as a single token, along with `"e "`, which glues the end of a word to the space after it. On a large corpus, the same counting also fills the vocabulary with tokens that mix letters and punctuation, like "dog.", "dog!" and "dog?", which are all variations of the same word. While this byte-level tokenizer is a good improvement, it does not split at word boundaries. The GPT tokenizer fixes this.

## Build the GPT BPE tokenizer

The GPT-2 paper<d-cite key="radford2019gpt2"></d-cite> (Section 2.2) observed exactly this problem: byte-level BPE wastes vocabulary slots on many versions of common words, like "dog.", "dog!" and "dog?". To avoid this, they prevent BPE from merging across character categories for any byte sequence, with an exception for spaces. In practice, GPT-2 and its successors first split the text into chunks with a regular expression, and then run BPE on each chunk separately, so merges never cross the boundary of a chunk. This step is called pre-tokenization.

We can extend our `ByteLevelTokenizer` to do this. The regex pattern below is the one used by the `cl100k_base` tokenizer of GPT-4 and GPT-3.5-turbo in tiktoken<d-cite key="openai2022tiktoken"></d-cite>, as written in minbpe<d-cite key="karpathy2024minbpe"></d-cite>. It uses `\p{L}` (any letter) and `\p{N}` (any number), which Python's built-in `re` module does not support, so we use the `regex` package (`pip install regex`).

```python
import regex

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class GPTLikeTokenizer(ByteLevelTokenizer):
    def __init__(self, default_merges: dict[tuple[int, int], int] = None):
        super().__init__(default_merges=default_merges)
        self.split_pattern = regex.compile(GPT4_SPLIT_PATTERN)

    def split_text(self, text: str) -> list[str]:
        return self.split_pattern.findall(text)

    def get_stats_multisegment(self, ids_list: list[list[int]]) -> dict[tuple[int, int], int]:
        token_stats = {}
        for ids in ids_list:
            for tk1, tk2 in zip(ids, ids[1:]):
                token_stats[(tk1, tk2)] = token_stats.get((tk1, tk2), 0) + 1
        return token_stats

    def merge_tokens_multisegment(self, ids_list: list[list[int]], ids: tuple[int, int], token: int) -> list[list[int]]:
        return [self.merge_tokens(sub_ids, ids, token) for sub_ids in ids_list]

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        # Split the text into chunks and encode each chunk separately into bytes
        text_list = self.split_text(text)
        ids_list = [self.byte_encode(t) for t in text_list]

        # First apply any pre-existing merges to our training tokens
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (tk1, tk2), tk_idx in sorted_merges:
            ids_list = self.merge_tokens_multisegment(ids_list, (tk1, tk2), tk_idx)

        total_iters = vocab_size - 256 - len(self.merges)

        for i in range(total_iters):
            stats = self.get_stats_multisegment(ids_list)
            if not stats:
                break
            tk1, tk2 = max(stats, key=stats.get)

            new_tk = 256 + len(self.merges)
            self.merges[(tk1, tk2)] = new_tk
            if verbose:
                print(f"Merging tokens {tk1} and {tk2} into token {new_tk}")
            ids_list = self.merge_tokens_multisegment(ids_list, (tk1, tk2), new_tk)

        self.vocab = self.get_vocab()

    def encode(self, text: str) -> list[int]:
        # Encode each chunk separately with the parent's encode, then concatenate
        final_tokens = []
        for t in self.split_text(text):
            final_tokens.extend(super().encode(t))
        return final_tokens
```

`get_stats_multisegment` counts pairs inside each chunk only, and `merge_tokens_multisegment` merges inside each chunk only. Everything else is the same as the byte-level tokenizer.

### The split pattern

The regex pattern is a list of alternatives separated by `|`. At each position in the text, the first alternative that matches is used, and the matched text becomes one chunk. From left to right;

- `'(?i:[sdmt]|ll|ve|re)` matches English contractions: `'s`, `'d`, `'m`, `'t`, `'ll`, `'ve` and `'re`, in upper or lower case.
- `[^\r\n\p{L}\p{N}]?+\p{L}+` matches a word, i.e. a run of letters, optionally preceded by one character that is not a newline, a letter or a number. Usually, this is a space, so `" world"` is one chunk. This is why many tokens in GPT vocabularies start with a space.
- `\p{N}{1,3}` matches numbers, at most 3 digits at a time.
- ` ?[^\s\p{L}\p{N}]++[\r\n]*` matches punctuation and symbols, with an optional leading space and any trailing newlines.
- `\s*[\r\n]` matches whitespace that ends with a newline.
- `\s+(?!\S)` matches whitespace, but leaves out the last space if a non-space character follows it, so that space can attach to the next word or punctuation.
- `\s+` matches any remaining whitespace.

Let's see it in action.

```python
gpt_tokenizer = GPTLikeTokenizer()
print(gpt_tokenizer.split_text("Hello've world123 how's   are you!!!?\n"))
print(gpt_tokenizer.split_text("1234567 tokens"))

# ['Hello', "'ve", ' world', '123', ' how', "'s", '  ', ' are', ' you', '!!!?\n']
# ['123', '456', '7', ' tokens']
```

Notice how "world" and "123" are separated, `'s` is split from "how", the extra spaces before "are" become their own chunk, and "1234567" is split into chunks of at most 3 digits.

The GPT-2 pattern, from its `encoder.py`<d-cite key="openai2019gpt2code"></d-cite>, was simpler: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`. Its contractions were case-sensitive, so "I'LL" was split differently from "I'll", a number of any length was a single chunk, and newlines were not handled separately.

Now, let's train our GPT-like tokenizer on the same Yorùbá text and sentence as before.

```python
gpt_tokenizer.train(yoruba_text, vocab_size=350)
ids = gpt_tokenizer.encode(sentence)
print(len(sentence.encode("utf-8")), "bytes ->", len(ids), "tokens")
print([gpt_tokenizer.decode([idx]) for idx in ids])
print(gpt_tokenizer.decode(ids) == sentence)

# The longest tokens learned during training
learned = [gpt_tokenizer.vocab[idx] for idx in range(256, 350)]
longest = sorted(learned, key=len, reverse=True)[:5]
print([tok.decode("utf-8", errors="replace") for tok in longest])

# 74 bytes -> 56 tokens
# ['E', ' n', 'i', 'l', 'o', ' l', 'a', 'ti', ' ', 'd', 'ọ', 'b', 'a', 'lẹ', '.', ' N', 'i', ' a', 's', 'a', ' ', 'w', 'a', ' a', 'ti', ' n', 'i', ' �', '�', 'd', 'è', ' ', 'w', 'a', ',', ' �', '�', 'w', 'ó', ' ', 's', 'e', ' p', 'a', 't', 'a', 'k', 'i', ' ', 'g', 'i', 'd', 'i', 'g', 'a', 'n']
# True
# [' Nàìjíríà', ' Nàìjírí', ' ìpínlẹ', ' Ìpínlẹ', ' orílẹ']
```

Compare these tokens with those of the byte-level tokenizer. Tokens like `"ti "`, `"i "`, `"è "` and `". "` that ended with a space are gone. Instead, the space is now at the start of the word, as in `" n"`, `" l"` and `" a"`, because the regex keeps the space with the word that follows it. `","` is also a separate token, since punctuation is split from letters. The longest learned tokens are now single words with their leading space, like `" Nàìjíríà"`, instead of phrases like `"ní orílẹ̀-èdè "`. Our sentence takes 56 tokens instead of 54, as merges can no longer cross word boundaries, but the vocabulary is spent on reusable words rather than on phrases.

We still see `" �"`, `"�"`, because the regex splits text by characters but BPE merges bytes, so a merge can still stop in the middle of a character. Also, notice that `" orílẹ"` and `" ìpínlẹ"` end just before a tone mark. In "orílẹ̀", the low tone on "ẹ" is a combining grave accent (U+0300), which Unicode does not classify as a letter (`\p{L}`) but as a mark (`\p{M}`). So the regex splits "orílẹ̀" into `" orílẹ"` and `"̀-"`. The GPT-4 pattern was designed with languages like English in mind, and splits words in languages that use combining marks, like Yorùbá, in the middle.

Lastly, let's go back to our default merges example, and train the GPT-like tokenizer on the same sentence.

```python
gpt_low_resource_tokenizer = GPTLikeTokenizer(default_merges=default_merges)
gpt_low_resource_tokenizer.train("This is how we see all these things", vocab_size=270)
ids = gpt_low_resource_tokenizer.encode("This is how we see it")
print([gpt_low_resource_tokenizer.decode([idx]) for idx in ids])

# ['This', ' is', ' how', ' we', ' ', 'se', 'e', ' i', 't']
```

The learned tokens now stay within the chunks of the regex split: `"This"`, `" is"` and `" how"` are whole words with their leading space, and no token runs into the next word.

## Where tokenization fails

We now have a tokenizer that works like the ones used in LLMs. Let's look at some of the places where tokenization, and BPE in particular, causes problems. Many of these show up as strange behaviours of the LLM, even though the model itself is not at fault.

- **Special tokens.** LLMs are trained on documents separated by special tokens like `<|endoftext|>` in GPT models. Our tokenizer would split this string into ordinary bytes. Real tokenizers add special tokens to the vocabulary with reserved IDs, and handle them before the regex split. They also have to decide what to do when a user types the special token string in a prompt.
- **Unknown characters.** A character-level BPE tokenizer cannot encode characters outside its base vocabulary, as we saw with "T". Byte-level BPE fixes this, but rare characters cost several tokens each.
- **Normalization.** The same visible text can be written with different code points, as we saw with "ọ̀". Without normalization, it is tokenized into different tokens, and the model has to learn that they mean the same thing.
- **Pre-tokenization.** The regex split decides which merges are possible. As we saw with "orílẹ̀", the GPT-4 pattern splits words that contain combining marks, so the tokenizer can never learn them as single tokens.
- **Unequal cost across languages.** A tokenizer trained mostly on English text learns long English tokens, while text in other languages, especially those with diacritics or non-Latin scripts, is split into many more tokens. Petrov et al.<d-cite key="petrov2023unfairness"></d-cite> found that the same text translated into different languages can differ in length by up to 15 times with the same tokenizer. This means these languages get a shorter effective context, and cost more to process.
- **Partial characters.** A token can hold an incomplete UTF-8 sequence. When an LLM generates text token by token, a single token may not decode to a valid string, so streaming applications have to buffer bytes until a character is complete.
- **Spelling, counting and arithmetic.** The model sees tokens, not characters. Counting the letters in a word, reversing a string or spelling a word requires knowledge that is hidden inside the tokens. Numbers are also split into chunks that do not line up with place value, e.g. "1234567" into "123", "456" and "7", which makes arithmetic harder to learn.
- **Whitespace.** Since the space is part of the word that follows it, `" world"` and `"world"` are different tokens. A trailing space at the end of a prompt leaves a lone space token that the model rarely saw during training, which can hurt its completions.
- **Untrained tokens.** The tokenizer can be trained on a different dataset than the model. Tokens that are frequent in the tokenizer's training data but almost absent from the model's training data likely get embeddings that are barely trained. These "glitch tokens", like `" SolidGoldMagikarp"` in the GPT-2 and GPT-3 vocabulary, made GPT-3 behave erratically when they appeared in a prompt<d-cite key="rumbelow2023solidgoldmagikarp"></d-cite>.

## Conclusion

In this article, we built a BPE tokenizer step by step, first on characters as described by Sennrich et al., then on UTF-8 bytes, and finally with the regex pre-tokenization used by GPT tokenizers. Byte pair encoding remains one of the most widely used tokenization methods in state-of-the-art LLMs, because it can encode any text with a small base vocabulary, while keeping sequences short. However, as we have seen, many issues with LLMs come from the tokenizer, so it is worth understanding how it works before blaming the model.

For further reading, Andrej Karpathy's video on building the GPT tokenizer<d-cite key="karpathy2024tokenizervideo"></d-cite> is a great companion to this article.
