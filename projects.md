---
layout: page
title: Projects
subtitle: Side projects and experiments. More on <a href="https://github.com/ogunlao">GitHub</a>.
---
{% for project in site.data.projects %}
<section class="project">
  {% if project.image %}
  <img src="{{ project.image | relative_url }}" alt="" loading="lazy"{% if project.fit %} style="object-fit: {{ project.fit }}"{% endif %}>
  {% else %}
  <div class="project-icon" aria-hidden="true">{{ project.icon }}</div>
  {% endif %}
  <div>
    <h2>{{ project.title }}</h2>
    <ul>
      {% for point in project.points %}<li>{{ point | markdownify | remove: "<p>" | remove: "</p>" }}</li>{% endfor %}
    </ul>
    {% if project.links %}
    <ul class="pub-links">
      {% for link in project.links %}<li><a href="{{ link.url | relative_url }}" target="_blank" rel="noopener">{{ link.name }}</a></li>{% endfor %}
    </ul>
    {% endif %}
  </div>
</section>
{% endfor %}
