(function () {
  var root = document.documentElement;

  // Light / dark toggle. The initial theme is set inline in <head>.
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) {}
      // Disqus picks its colour scheme on load; reload it so it matches.
      if (window.DISQUS) {
        window.DISQUS.reset({ reload: true, config: window.disqus_config });
      }
    });
  }

  // Back-to-top button.
  var topLink = document.getElementById("top-link");
  if (topLink) {
    var onScroll = function () {
      topLink.classList.toggle("visible", window.scrollY > 800);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
})();
