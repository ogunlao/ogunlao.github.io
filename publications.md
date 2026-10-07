---
layout: page
title: Publications
subtitle: Papers, theses and talks.
---
{% assign years = site.data.publications | group_by: "year" %}
{% for group in years %}
<h2 class="pub-year">{{ group.name }}</h2>
{% for pub in group.items %}
<div class="pub">
  <div class="pub-title">{{ pub.title }}</div>
  <div class="pub-authors">{{ pub.authors | replace: "Ogun, S.", "<strong>Ogun, S.</strong>" | replace: "Sewade Ogun", "<strong>Sewade Ogun</strong>" }}</div>
  <div class="pub-venue"><em>{{ pub.venue }}</em></div>
  {% if pub.links %}
  <ul class="pub-links">
    {% for link in pub.links %}<li><a href="{{ link.url | relative_url }}" target="_blank" rel="noopener">{{ link.name }}</a></li>{% endfor %}
  </ul>
  {% endif %}
</div>
{% endfor %}
{% endfor %}

<h2 class="pub-year" id="talks">Talks &amp; presentations</h2>
{% for talk in site.data.talks %}
<div class="pub">
  <div class="pub-title"><a href="{{ talk.url }}" target="_blank" rel="noopener">{{ talk.title }}</a></div>
  <div class="pub-venue">{{ talk.venue }}</div>
</div>
{% endfor %}
