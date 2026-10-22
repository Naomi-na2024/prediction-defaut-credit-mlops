// Petites animations : le score défile jusqu'à sa valeur et, sur téléphone,
// la page descend jusqu'au résultat. Sans JavaScript, tout reste lisible.
(function () {
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    document.querySelectorAll("[data-count]").forEach(function (el) {
        var end = parseInt(el.dataset.count, 10);
        var start = parseInt(el.dataset.from || "0", 10);
        if (reduce || isNaN(end)) return;
        var duration = 1100;
        var t0 = null;
        function step(now) {
            if (t0 === null) t0 = now;
            var p = Math.min((now - t0) / duration, 1);
            var eased = 1 - Math.pow(1 - p, 3);
            el.textContent = Math.round(start + (end - start) * eased);
            if (p < 1) requestAnimationFrame(step);
        }
        el.textContent = start;
        requestAnimationFrame(step);
    });

    var target = document.querySelector("[data-scroll-to]");
    if (target && window.innerWidth < 960) {
        target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
})();
