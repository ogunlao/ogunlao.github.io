---
layout: post
title:  "How to create a speech dataset for ASR, TTS, and other speech tasks"
categories: blog
tags: [speech, asr, tts]
comments: true
# categories: coding
# tags: linux

---

Over the past few months, I have come across a plethora of questions related to dataset creation for speech projects. I could not find a concise resource detailing all the necessary factors that need to be put in place to have a well balanced, unbiased and clean speech corpus. Many of the answers are distributed across research papers, online platforms and data repositories. As it is well known in the Machine Learning community, creating good datasets for predictive tasks require a ton of effort and attention to detail to get the right results. This article will report my findings on dataset creation for speech related tasks. It will be most useful for students, software engineers and researchers preparing to create their own corpus for specific tasks, especially in the low resource domain. The focus will be on creating corpus for Automatic Speech Recognition (ASR) but the ideas will still be useful for Text-To-Speech(TTS), Speech translation, Speaker classification and other machine learning tasks requiring speech as a modality.

Here's what we'll cover:
1. TOC
{:toc}

## Introduction

A speech corpus is a database containing audio recordings and the corresponding label. The label depends on the task. For ASR tasks, the label is the text, for TTS, the label is the audio itself, while the input is text. For speaker classification, the label will be the speaker id. Therefore, the label and data depends on the particular task. For ASR, the audio samples and text require that they correspond to the same entity. There is a large amount of recorded audio which can be sourced from podcasts, streaming platforms like Youtube, and even talk shows. As much as audio is available, there are major challenges in using them for speech task are that should be considered;

- they may contain artefacts/noise which are not important to the task and machine learning models may find it hard separating those artefacts from the actual signal,
- recordings may/may not have corresponding transcription which may be needed,
- multiple speakers talking simultaneously in the recordings,
- audio recordings may have to be split into short durations, with alignment performed with the corresponding text,
- Podcasts may have music playing at the background, multiple simultaneous speakers

With these problems in mind, you may need to determine if the audio is well suited for your task. In this article, we will focus on read speech for creating our own corpus, instead of relying on pre-recorded audio as explained above.

![Photo by Kate Oseen on Unsplash](/images/kate-oseen-XQKUIPjPl-s-unsplash.jpg)

# Getting Started

Since 2015, we have seen advances in using deep neural networks for ASR tasks [\[Papers with code\]](https://paperswithcode.com/sota/speech-recognition-on-librispeech-test-clean), surpassing previous models using ensembles of Gaussian Mixtures on various speech related task. Also, the introduction of the Connectionist Temporal Classification [\[A Graves, 2006\]](https://dl.acm.org/doi/10.1145/1143844.1143891) Loss has given a major boost to machine learning tasks like speech where alignment between the audio and text is cumbersome. Using the CTC Loss enables the model to maximize the objective over all possible and correct alignments between the audio and text. With these advancements, creating speech data has become significantly easier than previously imagined, with corpus requiring no alignment between the text and the read speech. 

For a detailed introduction to CTC Loss, checkout my blog post on [\[Breaking Down the CTC Loss\]](https://ogunlao.github.io/blog/2020/07/17/breaking-down-ctc-loss.html)

### Sampling Frequency

44.1kHz is the most common sampling frequency used to produce most digital audio. This ensures that the audio can be reconstructed for frequency below 22.05kHz, since it covers all frequencies that can be heard by a human. ASR experiments do not require that high sampling rate; more common frequencies are 8kHz and 16kHz, with sampling frequency of `16kHz` becoming the de facto in speech experiments, as there is no significant improvement in using a sampling frequency higher than that, but a lower sampling frequency may reduce accuracy. Therefore, it is advisable to stick to the 16kHz rule. Also, increasing the sampling frequency beyond that just increases the overhead during preprocessing and training, with some training procedures taking twice as much time, without any improvements.

On the converse, modern production quality TTS often use 24, 32, 44.1, or 48 kHz sampling rate, as 16kHz is too low to achieve high quality TTS [\[LibriTTS - Heiga Zen et al.\]](https://arxiv.org/pdf/1904.02882.pdf). For TTS, the acoustic model needs to learn the fine-grained acoustic characteristic of the audio to be able to reproduce the same form of signal from text.

During the signal preprocessing, the audio can be downsampled to its required sampling rate.

### Audio Format and Encoding

Audio format and encoding are two different things. Most popular file format used for speech-to-text experiments is the ".WAV" format. Since 'wav' is just a file format, it will have to be encoded during recording using one of the different encoding formats available such as Linear PCM Encoding.You do not need to worry about the details since this would be taken care of for you during your setup. Encodings can be lossy or lossless, taking up different file sizes and quality.

If your read speech corpus is saved in the MP3 file format, you may have to convert them to ".wav" during the preprocessing stage.

For a brief overview of encodings and audio formats, checkout the article [\[Introduction to audio encoding - GCP \]](https://cloud.google.com/speech-to-text/docs/encoding)

### Length of Recordings

For ASR tasks, the length of the audio samples should be smaller than about 30 seconds. Typically, for the ASR tasks I have worked on, the average length of the recordings range between 10 seconds and 15 seconds. The shorter the duration the better for the model, especially models that use recurrent neural networks (RNN) for decoding. There is the problem of long term dependency that needs to be addressed, especially in the low data regime and when using recurrent neural networks. It is also advisable to ensure the variance between the audio duration is small.

In the case of TTS, recordings should be splitted on sentence stops instead of silence intervals, to learn long-term characteristics of speech such as the sentence-level prosody for given a text [\[LibriTTS - Heiga Zen et al.\]](https://arxiv.org/pdf/1904.02882.pdf)

Other speech classification tasks such as gender identification and speaker identification, do not require long duration of samples. Typical duration of audio is 2  to 4 secs, which is enough to learn the signal characteristic for each class.

### Labels

As previously mentioned, the task determines the label for the audio. For example, Automatic Speech Translation (AST) requires text in the target language which may differ from the source language/language of the audio.

It is good practice to have a balanced sample-to-label ratio, with every label well represented. For instance, speaker identification tasks will require that the number of samples assigned to each speaker should be balanced. If one speaker is over represented, the acoustic model may learn unimportant characteristics of the speaker neglecting important signals. There are sampling methods though to forestall this situation, and some loss functions can be used to cater for the imbalance.

In the case of ASR tasks, the text should contain all alphabets of the target language in considerable proportion. Even for phoneme recognition tasks, all phones should be well represented in the labels. An example of a good phoneme recognition corpus is the [
TIMIT Acoustic-Phonetic Continuous Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1)

### Number of Speakers

The more the number of speakers, the better for the acoustic model, as it will also have to hear variations of speakers in the wild when deployed. It also ensures that we have a significant sample of speakers in the validation and test set.

## Annotator Characteristics

There are some characteristics of the speaker which are desirable for a balanced and unbiased data set. Some of these will be discussed here. The final task sometimes will determine where to focus on these characteristics. For example, if we can determine beforehand the target age group, we can easily focus on getting more data for them or even optimze for better predictions.

### Gender

The two gender groups (Male and Female) should be well represented in the data as the prosodic characteristics of males and females differ. It is ideal to have a 50-50 split in gender or close to it whenever possible.

### Age Groups

For general ASR tasks, all age groups should be represented, but may be difficult to accomplish for small ASR projects. Children under the age of 9 years, speak in a different way from adults. Their vocal characteristics begin to change at adolescence. All these should be put into consideration.

For audio recordings, not particular to humans, age may not be a requirement. For instance, in recordings involving animal sounds.

### Accents

Most cultures have moved across shores to other countries taking with them their language and tongue. The accents of those nations can affected how the language is spoken or communicated. For example, the Nigerian English differs significantly in pronunciations from the Indian English or the American English. Some production quality ASR learn different models for the different accent, but this is expensive. As humans, we easily adapt to accents after learning from a few examples in our environment.

Another idea may be to feed in an accent identifier into the acoustic model during training to adapt to different speaker accents.

### Other Metadata

Some metadata pertaining to the speaker should be collected during the recording. Speaker ID, age, country, text domain, Signal-to-Noise Ratio (SNR), time of recording etc. can be collected for each speaker. It is good practice to inform the annotators of the metadata that is been collected from them and how it will be used.

Also, depending on the task, these metadata can be used to properly sample from the corpus to avoid the imbalance we discussed earlier.

## Other important details to note

### Size of the data

As with all machine learning tasks involving deep neural networks, more data is better. The task can be split across many speakers to have a large sample size. Tasks like ASR and TTS require a lot of audio samples for good performance. The best models in English ASR are trained on about 60 thousand hours of speech [Jacob Kahn et al - LibriLight](https://arxiv.org/abs/1912.07875). That is equivalent to about 7 years of speech. This particular data was created from the LibriVox database of audio books.

In low resource audio settings, audio samples of this size may not be possible. We may then have to result to domain adaptation or training a self-supervised acoustic model from raw speech, if audio is available without transcriptions. There have been significant advances in unsupervised and self-supervised speech representation learnings enabling SOTA performance with limited data [Alexei Baevski et al. - wav2vec 2.0](https://arxiv.org/abs/2006.11477)

### Noise and Artefacts

Noise in all forms are a bane to good acoustic model performance, as they significantly affect the learning process. Significant research has been done to learn from noisy audio or noisy texts but it is still preferable to have clean text and well recorded audio. We need to ensure that the recording environment is devoid of background noise, music, animal sounds and even noise from electric devices such as Air conditioners.

Modern microphones and devices have noise filtering or noise cancelling mechanisms, giving better recording performance. It is a good idea to check if the recording device has this feature turned on. If affordable, recording studios can be created for the task.

On the converse, training with noisy audio can make the acoustic model robust to noise. The downstream task should determine the amount of noise permissible in the audio.

### Unlabelled audio

It may be difficult and more cumbersome to get large amounts of labelled data. Recent research has shown that clean unlabelled audio can also be useful for pretraining acoustic models. Unlabelled data may be easier to collect in many cases and that can be put to use in a self-supervised way. These methods have been shown to be competitive with their labelled counterparts for downstream tasks such as ASR. Two popular acoustic models for representation learning are [Aaron van den Oord et al - Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) and [A Baevski, 2020 - Wave2Vec2.0](https://arxiv.org/abs/2006.11477) are two popular unsupervised methods of learning speech representations for downstream tasks.

### Data Split

The examples should be splitted across the speakers. Speaker identity in the training set should not be represented in the validation and test set, and vice versa. This ensures that we can measure performance of the model on speakers and audio samples it has never seen during training.

The 80/10/10 rule of train/validation/test splitting can be applied when more data is available e.g 100 hrs. For low resources settings, this might be inapplicable, and therefore revert to ensuring that the test set is a good sample to test generalization. Given a total audio duration of 5 hours for instance, the audio can be split into 3/1/1 hours for train, validation and test respectively.

### Text Preprocessing

Typically for ASR tasks, the text may need to be cleaned and preprocessed to eliminate ambiguity in words and spellings. Digits can be spelled out in words where required, depending on the task. Also,in low resource settings, it is typical to convert all characters to their non-accented versions, reducing the character vocabulary size.

When preprocessing for ASR, punctuations are eliminated from the corpus as they are not typically read out during recitation, but just denoted with stops or gaps in the recording sequence. Words joined together by hyphens can be separated into two words. The apostrophe (\') character is left in the corpus for languages such as French, which use them for conjoining words.

In a similar vein to checking length of audio duration, the length of recorded text should be also be short when possible, to prevent errors in recordings. Long sentences can be split on words or stops. It is typical to use between 10 and 30 words for a single audio sample. The length should ensure that recordings do not exceed the 30 seconds mark as discussed above. All these help to prevent unnecessary gaps and stops or loss of attention while recording.

For ASR, a different text corpus from that used for recording will be needed to create an Language Model (LM). Language models are usually integrated into the decoding process of Speech-to-text systems for better performance. The performance gap decreases with the amount of training data though.

A language model with lower perplexity gives better decoding results. Transformers are becoming the de facto for modelling sequences such as text, and should be considered as the language model of choice if enough text is available to train it.

### Data Augmentation

Data Augmentation is an important technique in generating more data than available. For low resource settings, it is essential to augment the data with other versions of the audio recording. Augmentation can make the acoustic model less susceptible to overfitting.

Augmentation can be done on the raw speech or on the audio spectrogram. Some interesting augmentation methods like [SpecAugment](http://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) can be applied on the fly during preprocessing. It is a good practice to experiment with different augmentation strategies.

For a brief introduction to Audio augmentation, checkout the blog post by Edward Ma, [Data Augmentation for Audio](https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6)

## Recording details

This section talks about equipments and setup tools that may be used for audio annotation

### Recording tools

- Mobile phones: Modern mobile devices have very good microphones for recording sound. They can be paired with a recording application like [ligAikuma](https://lig-aikuma.imag.fr/) for annotation. Ensure you have a large storage space on the device to save recordings. One advantage of using mobile devices is that multiple annotators can record simultaneously and/or at their convenience, inasmuch as noise is being eliminated.

    The ligAikuma app is an app I recommend for recording, elicitation and translation. It was used in collecting my previous [speech project on Yoruba Language](https://github.com/ogunlao/yoruba_speech_project).

- Computer and Microphone: More sophisticated recording desktop applications are available for audio recordings. They can be paired to ensure noise-free recordings with great quality. The recording sampling rate, audio codec and audio format can be varied to give the desired output.

- Online recording platforms: There are also online recording platforms that do not even require any setup for recording. You can provide text and start recording almost immediately for free. Examples of such are [Common Voice platform](https://commonvoice.mozilla.org/en) and [Speech Annotation Toolkits for Low Resource Languages](https://www.dictate.app/). Ensure you read their SLA to determine how your data might be used by the platforms in the future.

In general, it is a good starting point to check online repositories like [Open SLR](http://openslr.org/index.html) and Common Voice for speech samples recorded by others. It gives a perspective on what to expect and how annotations should be done.

### Source of Text

Text is freely and openly available for high-resource languages like English, Mandarin, French etc. Some other languages of the world do not have large amount of text available for annotation. More often, texts are sourced from textbooks, news and media, religious publications e.g [JW.org](https://www.aclweb.org/anthology/P19-1310/) and the Bible. Wikipedia is also a good source of text for many languages and should be the first place to go, for clean text.

The acoustic model may be biased towards text from the specific domain it was trained on. So, care should be taken when using the acoustic model.

The most appropriate text is that which mimics the domain where the model will be used.

### Conclusion

This article explained in detail the various aspects of data collection that needs to be considered when creating a speech corpus, specifically for ASR. #

The article might require a re-read to digest every thing we have talked about. You can always have a reread at a later time. Also, feel free to initiate discussion in the comment section