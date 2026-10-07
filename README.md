# ogunlao.github.io

Source for [Sewade Ogun's blog](https://ogunlao.github.io/), built with Jekyll and served by GitHub Pages.

## Writing

- New post: add `_posts/YYYY-MM-DD-slug.md` with `title`, `tags` and optionally `description` (shown on the home page) in the front matter. Math (`$...$`, `$$...$$`), Disqus comments and reading time are on by default.
- Table of contents: put `1. TOC` followed by `{:toc}` where it should appear.
- News, talks, publications and projects live in `_data/*.yml`.

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
