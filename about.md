---
layout: page
title: About
permalink: /about/
redirect_from:
  - /about.html
  - /cv.html
---
I'm **Sewade Olaolu Ogun, PhD**. I build AI products at [GetVocal AI](https://getvocal.ai/) in Paris.

I did my PhD in Computer Science at [Inria Nancy](https://www.inria.fr/en/centre-inria-nancy-grand-est) and Vivoka, on generating diverse synthetic data for ASR training data augmentation. Before that, I completed the [African Masters in Machine Intelligence (AMMI)](https://aimsammi.org/) at AIMS.

This blog is where I write about what I'm learning, mostly through the lens of multimodality.

## Research interests

- Speech language models
- Generative text-to-speech systems
- Automatic speech recognition systems
- Dataset curation and augmentation
- Large language models

## CV

[Download my CV (PDF)](/archive/SewadeOgunCV.pdf) or see my [LinkedIn profile](https://www.linkedin.com/in/sewade-ogun/). Papers and talks are on the [Publications](/publications.html) page.

## News

<ul class="timeline">
{% for item in site.data.news %}
  <li><span class="date">{{ item.date | date: "%b %Y" }}</span><span>{{ item.text | markdownify | remove: "<p>" | remove: "</p>" }}</span></li>
{% endfor %}
</ul>

## Contact

Email: sogun [at] aimsammi [dot] org
