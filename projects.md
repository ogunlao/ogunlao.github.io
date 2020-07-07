---
layout: page
title: "Main Projects"
tagline : ""
use_math: true
lang: zh
---
{% include JB/setup %}

{% assign posts_collate = site.categories.projects %}
{% include JB/posts_collate %}

--- 

<link rel="stylesheet" href="/glyphicons/css/glyphicons.css" />

<table style="width:100%">
<col width="20%">
<col width="10">
<col >

<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![yoruba face](/images/project/yor_face.jpeg){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**Yoruba Speech Dataset: A dataset for low-resource speech tasks.**
- 3 hours of audio recordings from a native yoruba speaker.
- Can be used for low-resource speech tasks
- Details of the project can be found on my [github repo](https://github.com/ogunlao/yoruba_speech_project)

</td>
</tr>

</table>

<table style="width:100%">
<col width="20%">
<col width="10">
<col >


<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![yoruba face](/images/project/yor_face.jpeg){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**Comparing Yoruba and French Speech Models in Low-Resource Settings**
- CPC models is trained on 1 hour of yoruba and french audio with labels
- Pretrained CPC model from Facebook CPC_audio library is then finetuned on 1hr of data.
- Details of the project can be found on my [github repo](https://github.com/ogunlao/yoruba_speech_project/tree/master/project)

</td> 
</tr>
</table>

<table style="width:100%">
<col width="20%">
<col width="10">
<col >


<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![exploring detectron2](/images/project/object_tracking.jpg){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**Exploring Detectron2 for Object Detection Tasks**
- Instance segmentation, pose estimation, object tracking performed using the detectron2 library
- Pretrained models finetuned on custom tasks
- Details of the project can be found on my [github repo](https://github.com/ogunlao/exploring_detectron2)

|| <em class="icon-film"/> || [Object tracking demo](https://github.com/ogunlao/exploring_detectron2/blob/master/ObjectTracking/tracked_vid.avi) ||

</td> 
</tr>
</table>

<table style="width:100%">
<col width="20%">
<col width="10">
<col >

<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![exploring detectron2](/images/project/cassava_disease.png32){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**Cassava Disease Classification**
- Classify cassava plants as belonging to one of 4 various disease classes or healthy
- Experimented with various CNN architectures for this task
- Explore different methods of dealing with imbalanced datasets. I wrote a [blog post of what worked for me](https://ogunlao.github.io/2020/05/08/cross-validation-and-reproducibility-in-neural-networks.html)
- Details of the project can be found on my [github repo](https://github.com/ogunlao/cassava_disease_classification)

</td> 
</tr>
</table>


<table style="width:100%">
<col width="20%">
<col width="10">
<col >

<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![dna sequence](/images/project/dna_sequence.jpeg){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**DNA Sequence Classification using Kernel Methods**
- Predict whether DNA sequence region is binding site using Kernel methods
- Experimented with different kernels such as mismatch kernel, gaussian kernel, weighted-degree kernel
- Details of the project can be found on my [github repo](https://github.com/ogunlao/dna-sequence-kernels)

</td> 
</tr>
</table>

<table style="width:100%">
<col width="20%">
<col width="10">
<col >

<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
<!-- ![dog cat](/images/project/){:class="img-shadow"} -->
</td>
<td></td>
<td markdown="1">
**Dog Cat CLassifier**
- A simple ml pipeline for discriminating between dogs and cats
- Experiment can be found on my [github repo](https://github.com/ogunlao/dog-cat-classifier)

</td> 
</tr>
</table>

<table style="width:100%">
<col width="20%">
<col width="10">
<col >

<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![dog cat](/images/project/matching_game.png){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**The Matching Game**
- A game written in React/Javascript to test attention and retention
- Game hosted on netlify. You can play using the [game link](https://the-matching-game.netlify.com/)

</td> 
</tr>
</table>

<!-- <table style="width:100%">
<col width="20%">
<col width="10">
<col >


<tr height="25"/>
<tr style="border-bottom:1pt solid #eee" >
<td markdown="1">
![yoruba face](/images/project/yor_face.jpeg){:class="img-shadow"}
</td>
<td></td>
<td markdown="1">
**Comparing Yoruba and French Speech in Low-Resource Settings**
- CPC models is trained on 1 hour of yoruba and french audio with labels
- Pretrained CPC model from Facebook CPC_audio library is then finetuned on 1hr of data.
- Details of the project can be found on my [github repo](https://github.com/ogunlao/yoruba_speech_project/tree/master/project)

|| <em class="icon-film"/> || [video demo](https://youtu.be/XUTCowMHSQs) ||

</td> 
</tr>
</table> -->


<style type="text/css">
td {
    border: 0.5px;
    vertical-align: center;
    text-align: left;
}
</style>
