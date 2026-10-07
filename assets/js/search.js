(function () {
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  var posts = null;

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function highlight(text, terms) {
    var out = escapeHtml(text);
    terms.forEach(function (t) {
      var re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
      out = out.replace(re, "<mark>$1</mark>");
    });
    return out;
  }

  function snippet(content, term) {
    var i = content.toLowerCase().indexOf(term);
    if (i < 0) return content.slice(0, 180) + "…";
    var start = Math.max(0, i - 70);
    return (start > 0 ? "…" : "") + content.slice(start, start + 200) + "…";
  }

  function render(query) {
    var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { results.innerHTML = ""; return; }

    var scored = posts.map(function (p) {
      var title = p.title.toLowerCase();
      var tags = p.tags.join(" ").toLowerCase();
      var body = p.content.toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i];
        var hit = 0;
        if (title.indexOf(t) >= 0) hit += 10;
        if (tags.indexOf(t) >= 0) hit += 5;
        if (body.indexOf(t) >= 0) hit += 1;
        if (!hit) return null; // every term must match somewhere
        score += hit;
      }
      return { post: p, score: score };
    }).filter(Boolean).sort(function (a, b) { return b.score - a.score; });

    if (!scored.length) {
      results.innerHTML = "<li>No posts found.</li>";
      return;
    }
    results.innerHTML = scored.map(function (r) {
      var p = r.post;
      return '<li><a class="title" href="' + p.url + '">' + highlight(p.title, terms) + "</a>" +
        ' <small>&middot; ' + p.date + "</small>" +
        '<p class="snippet">' + highlight(snippet(p.content, terms[0]), terms) + "</p></li>";
    }).join("");
  }

  fetch(input.getAttribute("data-index"))
    .then(function (r) { return r.json(); })
    .then(function (data) {
      posts = data;
      var q = new URLSearchParams(location.search).get("q");
      if (q) { input.value = q; }
      render(input.value);
      input.addEventListener("input", function () { render(input.value); });
    });
})();
