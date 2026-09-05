// NeuroViral client runtime. Injected once via gr.Blocks(head=...), so it runs
// as a real document script. A MutationObserver initializes each player/compare
// island the moment Gradio injects it (first load AND on every value swap).
//
// Schema v2: timeline points have {t, a, b, global}; `channels` describes what
// a/b mean (label + color); `viz` picks the left panel ("brain" or "meter").

(function () {
  var DEFAULT_CH = { a: { label: "channel A", color: "#2dd4bf" },
                     b: { label: "channel B", color: "#fb923c" } };

  // ---------------- player island ----------------
  function initPlayer(root, P) {
    var V = P.video, tl = V.timeline, dur = V.duration_sec;
    var CH = V.channels || DEFAULT_CH;
    var ca = CH.a.color, cb = CH.b.color, la = CH.a.label, lb = CH.b.label;
    var isBrain = V.viz === "brain";

    var $ = function (s) { return root.querySelector(s); };
    var viz = $(".nv-brain"), bx = viz.getContext("2d");
    var vidc = $(".nv-vidcanvas"), vc = vidc.getContext("2d");
    var tlc = $(".nv-timeline"), tc = tlc.getContext("2d");
    var seek = $(".nv-seek"), playBtn = $(".nv-play"), timeEl = $(".nv-time");
    var videoEl = $(".nv-video");

    // dynamic panel header + readouts (labels/colors come from the data)
    $(".nv-vizhead").innerHTML = isBrain
      ? '◑ simulated brain response <span class="nv-sub">medial view · fsaverage5</span>'
      : '◱ engagement signal <span class="nv-sub">audiovisual salience</span>';
    $(".nv-readouts").innerHTML =
      '<div class="nv-ro"><span class="nv-dot" style="background:' + ca + '"></span>' + la + ' <b class="nv-aval">–</b></div>' +
      '<div class="nv-ro"><span class="nv-dot" style="background:' + cb + '"></span>' + lb + ' <b class="nv-bval">–</b></div>';
    var aval = $(".nv-aval"), bval = $(".nv-bval");

    function ext(key) { var mn = 1e9, mx = -1e9; tl.forEach(function (p) { mn = Math.min(mn, p[key]); mx = Math.max(mx, p[key]); }); return [mn, mx]; }
    var aE = ext("a"), bE = ext("b");
    var allMin = Math.min(aE[0], bE[0]), allMax = Math.max(aE[1], bE[1]);
    var norm = function (v, e) { return e[1] === e[0] ? 0.5 : (v - e[0]) / (e[1] - e[0]); };
    var peak = tl[0]; tl.forEach(function (p) { if (p.a > peak.a) peak = p; });

    function valAt(key, t) {
      if (t <= tl[0].t) return tl[0][key];
      if (t >= tl[tl.length - 1].t) return tl[tl.length - 1][key];
      for (var i = 1; i < tl.length; i++) {
        if (tl[i].t >= t) { var a = tl[i - 1], b = tl[i], f = (t - a.t) / (b.t - a.t); return a[key] + f * (b[key] - a[key]); }
      }
      return tl[tl.length - 1][key];
    }

    // master clock
    seek.max = dur;
    var useVideo = false, clock = { t: 0, playing: false }, last = performance.now();
    if (P.videoUrl) {
      videoEl.src = P.videoUrl;
      videoEl.addEventListener("loadeddata", function () { useVideo = true; videoEl.style.display = "block"; vidc.style.display = "none"; });
      videoEl.addEventListener("error", function () { useVideo = false; });
    }
    function getT() { return useVideo ? videoEl.currentTime : clock.t; }
    function setT(t) { t = Math.max(0, Math.min(dur, t)); if (useVideo) videoEl.currentTime = t; else clock.t = t; }
    function playing() { return useVideo ? !videoEl.paused : clock.playing; }
    function play() { if (getT() >= dur - 0.02) setT(0); last = performance.now(); if (useVideo) videoEl.play(); else clock.playing = true; playBtn.textContent = "❚❚ pause"; }
    function pause() { if (useVideo) videoEl.pause(); else clock.playing = false; playBtn.textContent = "▶ play"; }
    playBtn.onclick = function () { playing() ? pause() : play(); };
    seek.oninput = function () { setT(parseFloat(seek.value)); draw(getT()); };

    function drawVidPlaceholder(t) {
      var w = vidc.width, h = vidc.height, g = vc.createLinearGradient(0, 0, w, h), ph = t / Math.max(dur, 1);
      g.addColorStop(0, "hsl(" + (200 + ph * 120) + ",55%,22%)");
      g.addColorStop(1, "hsl(" + (260 + ph * 120) + ",60%,14%)");
      vc.fillStyle = g; vc.fillRect(0, 0, w, h);
      vc.strokeStyle = "rgba(255,255,255,.10)"; vc.lineWidth = 2;
      for (var k = 0; k < 4; k++) { vc.beginPath(); var r = 30 + ((t * 40 + k * 60) % w); vc.arc(w * 0.5, h * 0.5, r, 0, 7); vc.stroke(); }
      vc.fillStyle = "rgba(255,255,255,.9)"; vc.font = "600 13px monospace"; vc.fillText(t.toFixed(1) + "s", 14, 26);
      vc.fillStyle = "rgba(255,255,255,.45)"; vc.font = "11px sans-serif"; vc.fillText("upload plays here · sample clips use a placeholder", 14, h - 14);
      if (Math.abs(t - peak.t) < 0.6) { vc.fillStyle = ca; vc.fillRect(w - 88, 14, 74, 20); vc.fillStyle = "#06121a"; vc.font = "600 11px sans-serif"; vc.fillText("PEAK", w - 74, 28); }
    }

    // --- brain viz (TRIBE mode) ---
    function brainOutline(g) {
      var w = viz.width, h = viz.height; g.save(); g.translate(w * 0.5, h * 0.52); g.scale(1.5, 1.5);
      g.beginPath(); g.moveTo(-140, 0);
      g.bezierCurveTo(-150, -70, -70, -100, 10, -92);
      g.bezierCurveTo(90, -84, 150, -40, 150, 0);
      g.bezierCurveTo(150, 35, 120, 60, 60, 64);
      g.bezierCurveTo(30, 66, 20, 58, 0, 60);
      g.bezierCurveTo(-40, 64, -60, 58, -90, 54);
      g.bezierCurveTo(-125, 48, -133, 30, -140, 0);
      g.closePath(); g.restore();
    }
    function hexrgb(h) { h = h.replace("#", ""); return [parseInt(h.substr(0, 2), 16), parseInt(h.substr(2, 2), 16), parseInt(h.substr(4, 2), 16)].join(","); }
    function glow(g, x, y, r, hex, alpha) {
      var rg = g.createRadialGradient(x, y, 0, x, y, r), c = hexrgb(hex);
      rg.addColorStop(0, "rgba(" + c + "," + alpha + ")"); rg.addColorStop(1, "rgba(" + c + ",0)");
      g.fillStyle = rg; g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
    }
    function drawBrain(t) {
      var w = viz.width, h = viz.height; bx.clearRect(0, 0, w, h);
      bx.fillStyle = "#0b1020"; bx.fillRect(0, 0, w, h);
      var gv = norm(valAt("global", t), [allMin, allMax]);
      bx.save(); brainOutline(bx); bx.fillStyle = "#1c2740"; bx.fill();
      bx.lineWidth = 2; bx.strokeStyle = "#33406a"; bx.stroke();
      brainOutline(bx); bx.clip();
      bx.fillStyle = "rgba(120,150,255," + (0.05 + 0.15 * gv) + ")"; bx.fillRect(0, 0, w, h);
      var av = norm(valAt("a", t), aE), bv = norm(valAt("b", t), bE);
      glow(bx, w * 0.72, h * 0.66, 60 + 40 * av, ca, 0.25 + 0.6 * av);
      glow(bx, w * 0.50, h * 0.52, 45 + 30 * bv, cb, 0.22 + 0.6 * bv);
      bx.restore();
      bx.fillStyle = "rgba(200,220,255,.75)"; bx.font = "11px sans-serif";
      bx.fillText(la, w * 0.66, h * 0.84); bx.fillText(lb, w * 0.40, h * 0.30);
    }

    // --- meter viz (audiovisual mode) ---
    function bar(x, y, w, v, color) {
      bx.fillStyle = "#1c2740"; bx.fillRect(x, y, w, 10);
      bx.fillStyle = color; bx.fillRect(x, y, w * v, 10);
    }
    function drawMeter(t) {
      var w = viz.width, h = viz.height; bx.clearRect(0, 0, w, h);
      bx.fillStyle = "#0b1020"; bx.fillRect(0, 0, w, h);
      var gv = Math.max(0, Math.min(1, norm(valAt("global", t), [allMin, allMax])));
      var cx = w * 0.5, cy = h * 0.42, r = Math.min(w, h) * 0.26;
      var s0 = Math.PI * 0.75, s1 = Math.PI * 2.25;
      bx.lineWidth = 16; bx.lineCap = "round";
      bx.strokeStyle = "#1c2740"; bx.beginPath(); bx.arc(cx, cy, r, s0, s1); bx.stroke();
      var grad = bx.createLinearGradient(cx - r, cy, cx + r, cy);
      grad.addColorStop(0, ca); grad.addColorStop(1, cb);
      bx.strokeStyle = grad; bx.beginPath(); bx.arc(cx, cy, r, s0, s0 + (s1 - s0) * gv); bx.stroke();
      bx.textAlign = "center";
      bx.fillStyle = "#fff"; bx.font = "700 36px sans-serif"; bx.fillText(Math.round(gv * 100), cx, cy + 8);
      bx.fillStyle = "#8fa0c4"; bx.font = "10px sans-serif"; bx.fillText("ENGAGEMENT NOW", cx, cy + 26);
      bx.textAlign = "left";
      bar(w * 0.15, h * 0.80, w * 0.7, Math.max(0, Math.min(1, norm(valAt("a", t), aE))), ca);
      bar(w * 0.15, h * 0.80 + 20, w * 0.7, Math.max(0, Math.min(1, norm(valAt("b", t), bE))), cb);
    }

    function drawTimeline(t) {
      var w = tlc.width, h = tlc.height, pl = 44, pr = 16, pt = 14, pb = 26;
      tc.clearRect(0, 0, w, h);
      var x = function (tt) { return pl + (tt / dur) * (w - pl - pr); };
      var y = function (v) { return pt + (1 - (v - allMin) / (allMax - allMin || 1)) * (h - pt - pb); };
      if (allMin < 0 && allMax > 0) { tc.strokeStyle = "#26304a"; tc.lineWidth = 1; tc.beginPath(); tc.moveTo(pl, y(0)); tc.lineTo(w - pr, y(0)); tc.stroke(); }
      tc.fillStyle = "#5f6f92"; tc.font = "10px sans-serif";
      for (var s = 0; s <= dur; s += Math.max(2, Math.round(dur / 6))) { tc.fillText(s + "s", x(s) - 6, h - 8); }
      function line(key, color) { tc.strokeStyle = color; tc.lineWidth = 2; tc.beginPath(); tl.forEach(function (p, i) { var px = x(p.t), py = y(p[key]); i ? tc.lineTo(px, py) : tc.moveTo(px, py); }); tc.stroke(); }
      line("b", cb); line("a", ca);
      var pkx = x(peak.t), pky = y(peak.a);
      tc.fillStyle = "#fde68a"; tc.beginPath();
      for (var i2 = 0; i2 < 5; i2++) { var ang = -Math.PI / 2 + i2 * 2 * Math.PI / 5; tc.lineTo(pkx + 7 * Math.cos(ang), pky + 7 * Math.sin(ang)); tc.lineTo(pkx + 3 * Math.cos(ang + Math.PI / 5), pky + 3 * Math.sin(ang + Math.PI / 5)); }
      tc.closePath(); tc.fill();
      tc.fillStyle = "#fde68a"; tc.font = "10px sans-serif"; tc.fillText("peak @" + peak.t.toFixed(0) + "s", pkx + 10, pky - 6);
      tc.fillStyle = ca; tc.fillText(la, w - 200, pt + 2); tc.fillStyle = cb; tc.fillText(lb, w - 110, pt + 2);
      var hx = x(t); tc.strokeStyle = "rgba(255,255,255,.85)"; tc.lineWidth = 1.5; tc.beginPath(); tc.moveTo(hx, pt); tc.lineTo(hx, h - pb); tc.stroke();
      tc.fillStyle = ca; tc.beginPath(); tc.arc(hx, y(valAt("a", t)), 3.5, 0, 7); tc.fill();
      tc.fillStyle = cb; tc.beginPath(); tc.arc(hx, y(valAt("b", t)), 3.5, 0, 7); tc.fill();
    }

    function draw(t) {
      if (!useVideo) drawVidPlaceholder(t);
      if (isBrain) drawBrain(t); else drawMeter(t);
      drawTimeline(t);
      aval.textContent = valAt("a", t).toFixed(2);
      bval.textContent = valAt("b", t).toFixed(2);
      seek.value = t; timeEl.textContent = t.toFixed(1) + " / " + dur.toFixed(1) + "s";
    }

    // score / drivers panel
    var sp = $(".nv-scorepanel");
    var drivers = (V.drivers && V.drivers.length) ? V.drivers
      : Object.keys(V.features || {}).map(function (k) { return [k, V.features[k]]; });
    var driverRows = function () {
      var mx = Math.max.apply(null, drivers.map(function (d) { return Math.abs(d[1]); })) || 1;
      return drivers.map(function (d) {
        var pctw = Math.round(Math.abs(d[1]) / mx * 100);
        return '<div class="nv-drow"><span class="nv-dname">' + d[0] + '</span>' +
          '<span class="nv-dbar"><span class="nv-dfill" style="width:' + pctw + '%"></span></span>' +
          '<span class="nv-dval">' + (+d[1]).toFixed(3) + "</span></div>";
      }).join("");
    };
    if (!V.prediction) {
      sp.style.gridTemplateColumns = "1fr";
      sp.innerHTML =
        '<div><div class="nv-scorelbl" style="margin-bottom:8px">what the signal shows</div>' +
        '<div class="nv-drivers">' + driverRows() + "</div>" +
        '<div class="nv-noscore" style="margin-top:10px">No trained engagement score — these are measured signals from the video, not a prediction. The score panel appears only when a validated model is available.</div></div>';
    } else {
      sp.innerHTML =
        '<div><div class="nv-scorebig">' + V.prediction.score.toFixed(3) + "</div>" +
        '<div class="nv-scorelbl">predicted engagement rate</div>' +
        '<div class="nv-scorelbl" style="margin-top:8px">rank <span class="nv-pct">' + V.prediction.percentile + "th pct</span></div></div>" +
        '<div><div class="nv-scorelbl" style="margin-bottom:8px">what drove it</div><div class="nv-drivers">' + driverRows() + "</div></div>";
    }

    // clock loop: setInterval (not rAF) so it keeps advancing even when the tab
    // is throttled/backgrounded — makes the demo robust on any machine.
    var timer = setInterval(function () {
      if (!root.isConnected) { clearInterval(timer); return; }
      var now = performance.now(), dt = (now - last) / 1000; last = now;
      if (!useVideo && clock.playing) { clock.t += dt; if (clock.t >= dur) { clock.t = dur; pause(); } }
      draw(getT());
    }, 33);
    draw(0);
  }

  // ---------------- compare island ----------------
  function initCompare(root, P) {
    var cols = root.querySelectorAll(".nvc-col"), data = [["hit", P.hit], ["flop", P.flop]];
    var mn = 1e9, mx = -1e9;
    data.forEach(function (d) { d[1].timeline.forEach(function (p) { mn = Math.min(mn, p.a, p.b); mx = Math.max(mx, p.a, p.b); }); });
    data.forEach(function (d, idx) {
      var V = d[1], tl = V.timeline, dur = V.duration_sec, col = cols[idx], c = col.querySelector(".nvc-c"), g = c.getContext("2d");
      var CH = V.channels || DEFAULT_CH, ca = CH.a.color, cb = CH.b.color;
      var w = c.width, h = c.height, pl = 34, pr = 10, pt = 10, pb = 22;
      var x = function (t) { return pl + (t / dur) * (w - pl - pr); }, y = function (v) { return pt + (1 - (v - mn) / (mx - mn || 1)) * (h - pt - pb); };
      g.clearRect(0, 0, w, h);
      if (mn < 0 && mx > 0) { g.strokeStyle = "#26304a"; g.beginPath(); g.moveTo(pl, y(0)); g.lineTo(w - pr, y(0)); g.stroke(); }
      function line(key, color) { g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); tl.forEach(function (p, i) { var px = x(p.t), py = y(p[key]); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); }
      line("b", cb); line("a", ca);
      var peak = tl[0]; tl.forEach(function (p) { if (p.a > peak.a) peak = p; });
      g.fillStyle = "#fde68a"; g.beginPath(); g.arc(x(peak.t), y(peak.a), 4, 0, 7); g.fill();
      g.fillStyle = "#5f6f92"; g.font = "10px sans-serif"; g.fillText("0s", pl - 4, h - 8); g.fillText(dur.toFixed(0) + "s", w - pr - 14, h - 8);
      var f = V.features || {}, e = V.engagement, pr2 = V.prediction;
      col.querySelector(".nvc-meta").innerHTML =
        "<b>" + V.title + "</b><br>" +
        "hook (3s): <b>" + (f.hook_3s != null ? f.hook_3s.toFixed(2) : "–") + "</b>" +
        (e && e.rate ? " · engagement rate: <b>" + (e.rate * 100).toFixed(2) + "%</b>" : "") +
        (pr2 ? " · predicted: <b>" + pr2.score.toFixed(3) + "</b> (" + pr2.percentile + "th)" : "");
    });
  }

  // ---------------- observer ----------------
  function payload(root, cls) {
    var el = root.querySelector("script." + cls);
    if (!el) return null;
    try { return JSON.parse(el.textContent); } catch (e) { console.error("nv payload parse", e); return null; }
  }
  function scan() {
    document.querySelectorAll(".nv-wrap:not([data-nvinit])").forEach(function (root) {
      var P = payload(root, "nv-payload"); if (!P) return;
      root.setAttribute("data-nvinit", "1");
      try { initPlayer(root, P); } catch (e) { console.error("nv init", e); }
    });
    document.querySelectorAll(".nvc-wrap:not([data-nvinit])").forEach(function (root) {
      var P = payload(root, "nvc-payload"); if (!P) return;
      root.setAttribute("data-nvinit", "1");
      try { initCompare(root, P); } catch (e) { console.error("nvc init", e); }
    });
  }
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });
  if (document.readyState !== "loading") scan();
  else document.addEventListener("DOMContentLoaded", scan);
})();
