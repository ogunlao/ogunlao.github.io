# ogunlao.github.io

Source for [Sewade Ogun's blog](https://ogunlao.github.io/), built with Jekyll and served by GitHub Pages.

## Writing

- New post: add `_posts/YYYY-MM-DD-slug.md` with `title`, `tags` and optionally `description` (shown under the title and on the home page) in the front matter. Posts use the Distill layout; Disqus comments, math and reading time are on by default.
- Add `hidden: true` to a post to keep it off the home page; it stays in the archive and search.
- Table of contents: put `1. TOC` followed by `{:toc}` where it should appear.
- News, talks, publications and projects live in `_data/*.yml`.

### Distill-style posts

Set `layout: distill` in a post's front matter to get the [Distill](https://distill.pub/) look: a wide title block, byline, hover citations and footnotes, a reference list and a "cite this post" block. See `_posts/2020-07-17-breaking-down-ctc-loss.md` for an example.

```yaml
layout: distill
title: "My post"
description: One-line summary shown under the title.
bibliography: my-post.bib   # file in assets/bibliography/
authors:                    # optional; defaults to the site author
  - name: Sewade Ogun
    url: https://ogunlao.github.io/about/
    affiliation: GetVocal AI
```

- Cite with `<d-cite key="bibtexkey"></d-cite>` (several keys: `key="a,b"`).
- Footnotes: `<d-footnote>Text shown on hover.</d-footnote>`.
- Wider figures: wrap them in `<div class="l-page">…</div>` (or `l-screen` for full width).
- Math is rendered by MathJax. Write it with `$$...$$`, which kramdown passes through untouched:
  inline inside a sentence (`the loss $$\mathcal{L}$$ is…`), and as a display equation when `$$` sits on its own lines.
  Avoid single `$...$`: Markdown then mangles `\{`, `\|`, `*`, `...` and quotes inside the math.
  Use `\begin{aligned} … \end{aligned}` inside a display block for derivations, and `\operatorname{name}` / `\text{word}` for multi-letter names.

The Distill template is vendored in `assets/js/distill/template.v2.js` (Apache-2.0).

## Running locally

```sh
bundle install
bundle exec jekyll serve --livereload
```

Or without a local Ruby:

```sh
docker run --rm -it -p 4000:4000 -v "$PWD":/srv/site -w /srv/site ruby:3.3 \
  bash -c "bundle install && bundle exec jekyll serve --host 0.0.0.0"
```

## Analytics

Set `google_analytics` in `_config.yml` to your GA4 measurement ID (`G-XXXXXXXXXX`). It is only rendered in production builds.
