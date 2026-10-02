/* Tiny dependency-free SVG chart renderer for the figures built in analyzer/charts.py.
   Kinds: line, area, bar, hbar, pie, scatter, histogram.
   Colours come from CSS variables --s1..--s8 so light/dark mode just works. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const color = (slot) => css(`--s${(slot % 8) + 1}`);

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  }

  // ---------- formatting ----------
  function compact(v) {
    if (v === null || v === undefined || isNaN(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e12) return (v / 1e12).toFixed(a >= 1e13 ? 0 : 1).replace(/\.0$/, "") + "T";
    if (a >= 1e9) return (v / 1e9).toFixed(a >= 1e10 ? 0 : 1).replace(/\.0$/, "") + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1).replace(/\.0$/, "") + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1).replace(/\.0$/, "") + "K";
    return Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/\.?0+$/, "");
  }
  const full = (v) => (v === null || v === undefined ? "–" :
    Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 }));
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  function dateLabel(s, grain) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
    if (!m) return s;
    if (grain === "year") return m[1];
    if (grain === "quarter") return `Q${Math.floor((+m[2] - 1) / 3) + 1} ${m[1]}`;
    if (grain === "month") return `${MONTHS[+m[2] - 1]} ${m[1].slice(2)}`;
    return `${+m[3]} ${MONTHS[+m[2] - 1]}`;
  }
  const trunc = (s, n) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));

  function niceTicks(min, max, count = 5) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = (span / count) / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
    const ticks = [];
    for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  // ---------- tooltip ----------
  function tooltip(container) {
    let t = container.querySelector(".viz-tip");
    if (!t) { t = document.createElement("div"); t.className = "viz-tip"; container.appendChild(t); }
    return {
      show(html, x, y) {
        t.innerHTML = html; t.style.display = "block";
        const w = t.offsetWidth, cw = container.clientWidth;
        t.style.left = Math.min(Math.max(x + 12, 0), cw - w - 4) + "px";
        t.style.top = Math.max(y - t.offsetHeight - 10, 0) + "px";
      },
      hide() { t.style.display = "none"; },
    };
  }
  const swatch = (slot) => `<i style="background:${color(slot)}"></i>`;

  function legend(container, series) {
    if (series.length < 2) return;
    const l = document.createElement("div");
    l.className = "viz-legend";
    l.innerHTML = series.map((s) => `<span>${swatch(s.slot)}${s.name}</span>`).join("");
    container.appendChild(l);
  }

  // ---------- cartesian frame ----------
  function frame(container, fig, opts) {
    const W = container.clientWidth || 600, H = opts.height || 280;
    const m = { t: 12, r: 16, b: 46, l: opts.left || 56 };
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "viz", role: "img" });
    container.appendChild(svg);
    return { svg, W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
  }

  function yAxis(f, ticks, scale, label) {
    const g = el("g", {}, f.svg);
    ticks.forEach((v) => {
      const y = scale(v);
      el("line", { x1: f.m.l, x2: f.W - f.m.r, y1: y, y2: y, class: v === 0 ? "axis" : "grid" }, g);
      el("text", { x: f.m.l - 8, y: y + 4, "text-anchor": "end", class: "tick" }, g).textContent = compact(v);
    });
    if (label) el("text", { x: 12, y: f.m.t + f.ih / 2, transform: `rotate(-90 12 ${f.m.t + f.ih / 2})`,
      "text-anchor": "middle", class: "axis-label" }, g).textContent = label;
  }

  function xCatAxis(f, labels, xpos, label) {
    const g = el("g", {}, f.svg);
    const maxLabels = Math.max(2, Math.floor(f.iw / 70));
    const every = Math.ceil(labels.length / maxLabels);
    labels.forEach((s, i) => {
      if (i % every !== 0 && i !== labels.length - 1) return;
      if (i === labels.length - 1 && i % every !== 0 && labels.length - 1 - (Math.floor(i / every) * every) < every / 2) return;
      el("text", { x: xpos(i), y: f.H - f.m.b + 18, "text-anchor": "middle", class: "tick" }, g).textContent = trunc(s, 14);
    });
    if (label) el("text", { x: f.m.l + f.iw / 2, y: f.H - 6, "text-anchor": "middle", class: "axis-label" }, g).textContent = label;
  }

  function barPath(x, y, w, h, r, horizontal) {
    // rounded only at the data end, square at the baseline
    r = Math.min(r, (horizontal ? h : w) / 2, horizontal ? w : h);
    if (r <= 0 || (horizontal ? w : h) <= 0) return `M${x},${y}h${w}v${h}h${-w}Z`;
    if (horizontal) return `M${x},${y}h${w - r}q${r},0 ${r},${r}v${h - 2 * r}q0,${r} ${-r},${r}h${-(w - r)}Z`;
    return `M${x},${y + h}v${-(h - r)}q0,${-r} ${r},${-r}h${w - 2 * r}q${r},0 ${r},${r}v${h - r}Z`;
  }

  // ---------- kinds ----------
  function lineChart(container, fig) {
    const grain = (fig.x_label.match(/\((\w+)\)$/) || [])[1];
    const labels = fig.categories.map((c) => dateLabel(c, grain));
    legend(container, fig.series);
    const f = frame(container, fig, {});
    const all = fig.series.flatMap((s) => s.values).filter((v) => v !== null);
    const ticks = niceTicks(Math.min(0, ...all), Math.max(...all));
    const ys = (v) => f.m.t + f.ih - ((v - ticks[0]) / (ticks.at(-1) - ticks[0])) * f.ih;
    const n = labels.length;
    const xs = (i) => f.m.l + (n === 1 ? f.iw / 2 : (i / (n - 1)) * f.iw);
    yAxis(f, ticks, ys, fig.y_label);
    xCatAxis(f, labels, xs, fig.x_label);

    // shade incomplete first/last periods underneath the line
    if (fig.partial_first && n > 1) {
      el("rect", { x: f.m.l, y: f.m.t, width: (xs(0) + xs(1)) / 2 - f.m.l, height: f.ih, class: "partial" }, f.svg);
    }
    if (fig.partial_last && n > 1) {
      el("rect", { x: (xs(n - 2) + xs(n - 1)) / 2, y: f.m.t, width: f.W - f.m.r - (xs(n - 2) + xs(n - 1)) / 2,
        height: f.ih, class: "partial" }, f.svg);
      el("text", { x: f.W - f.m.r - 4, y: f.m.t + 12, "text-anchor": "end", class: "tick" }, f.svg).textContent = "partial";
    }
    fig.series.forEach((s) => {
      const pts = s.values.map((v, i) => (v === null ? null : [xs(i), ys(v)]));
      const d = pts.reduce((acc, p, i) => (p ? acc + (acc && pts[i - 1] ? "L" : "M") + p[0] + "," + p[1] : acc), "");
      if (fig.kind === "area" && fig.series.length === 1)
        el("path", { d: d + `L${xs(n - 1)},${ys(Math.max(0, ticks[0]))}L${xs(0)},${ys(Math.max(0, ticks[0]))}Z`,
          fill: color(s.slot), "fill-opacity": 0.14, stroke: "none" }, f.svg);
      el("path", { d, fill: "none", stroke: color(s.slot), "stroke-width": 2, "stroke-linejoin": "round",
        "stroke-linecap": "round", "stroke-dasharray": null }, f.svg);
      if (n <= 40) pts.forEach((p) => p && el("circle", { cx: p[0], cy: p[1], r: 3.5, fill: color(s.slot),
        stroke: css("--surface"), "stroke-width": 1.5 }, f.svg));
    });

    // crosshair + tooltip
    const tip = tooltip(container);
    const cross = el("line", { y1: f.m.t, y2: f.m.t + f.ih, class: "cross", visibility: "hidden" }, f.svg);
    const hit = el("rect", { x: f.m.l, y: f.m.t, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
    hit.addEventListener("mousemove", (e) => {
      const r = f.svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - f.m.l) / f.iw) * (n - 1))));
      cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i)); cross.setAttribute("visibility", "visible");
      const rows = fig.series.map((s) => [s, s.values[i]]).sort((a, b) => (b[1] ?? -Infinity) - (a[1] ?? -Infinity));
      const partialHere = (fig.partial_last && i === n - 1) || (fig.partial_first && i === 0);
      tip.show(`<b>${labels[i]}${partialHere ? " (partial period)" : ""}</b>` +
        rows.map(([s, v]) => `<div>${swatch(s.slot)}${fig.series.length > 1 ? s.name + ": " : ""}<b>${full(v)}</b></div>`).join(""),
        xs(i), e.clientY - container.getBoundingClientRect().top);
    });
    hit.addEventListener("mouseleave", () => { tip.hide(); cross.setAttribute("visibility", "hidden"); });
  }

  function barChart(container, fig) {
    const horizontal = fig.kind === "hbar";
    const cats = fig.categories, S = fig.series;
    legend(container, S);
    const longest = Math.max(...cats.map((c) => String(c).length));
    const height = horizontal ? Math.max(220, cats.length * (S.length > 1 ? 14 * S.length + 10 : 26) + 60) : 280;
    const f = frame(container, fig, { height, left: horizontal ? Math.min(150, 14 + longest * 6.5) : 56 });
    const all = S.flatMap((s) => s.values);
    const ticks = niceTicks(Math.min(0, ...all), Math.max(...all));
    const span = ticks.at(-1) - ticks[0];
    const tip = tooltip(container);
    const gap = 2;

    if (!horizontal) {
      const ys = (v) => f.m.t + f.ih - ((v - ticks[0]) / span) * f.ih;
      yAxis(f, ticks, ys, fig.y_label);
      const band = f.iw / cats.length, inner = Math.min(band * 0.72, 22 * S.length + gap * (S.length - 1));
      const bw = (inner - gap * (S.length - 1)) / S.length;
      const xpos = (i) => f.m.l + band * i + band / 2;
      xCatAxis(f, fig.kind === "histogram" ? cats.map((c) => c.split("–")[0]) : cats, xpos, fig.x_label);
      cats.forEach((c, i) => S.forEach((s, j) => {
        const v = s.values[i], x0 = xpos(i) - inner / 2 + j * (bw + gap);
        const y0 = ys(Math.max(0, v)), h = Math.abs(ys(v) - ys(0));
        const bar = el("path", { d: barPath(x0, v >= 0 ? y0 : ys(0), bw, h, 4, false), fill: color(s.slot), class: "mark" }, f.svg);
        bar.addEventListener("mousemove", (e) => tip.show(`<b>${c}</b><div>${swatch(s.slot)}${S.length > 1 ? s.name + ": " : ""}<b>${full(v)}</b></div>`,
          x0 + bw / 2, e.clientY - container.getBoundingClientRect().top));
        bar.addEventListener("mouseleave", tip.hide);
      }));
    } else {
      const xs = (v) => f.m.l + ((v - ticks[0]) / span) * f.iw;
      const g = el("g", {}, f.svg);
      ticks.forEach((v) => {
        el("line", { x1: xs(v), x2: xs(v), y1: f.m.t, y2: f.m.t + f.ih, class: v === 0 ? "axis" : "grid" }, g);
        el("text", { x: xs(v), y: f.H - f.m.b + 18, "text-anchor": "middle", class: "tick" }, g).textContent = compact(v);
      });
      el("text", { x: f.m.l + f.iw / 2, y: f.H - 6, "text-anchor": "middle", class: "axis-label" }, g).textContent = fig.y_label;
      const band = f.ih / cats.length, inner = Math.min(band * 0.75, 18 * S.length + gap * (S.length - 1));
      const bh = (inner - gap * (S.length - 1)) / S.length;
      cats.forEach((c, i) => {
        const yc = f.m.t + band * i + band / 2;
        el("text", { x: f.m.l - 8, y: yc + 4, "text-anchor": "end", class: "tick cat" }, g).textContent = trunc(c, 22);
        S.forEach((s, j) => {
          const v = s.values[i], y0 = yc - inner / 2 + j * (bh + gap);
          const x0 = xs(Math.min(0, v)), w = Math.abs(xs(v) - xs(0));
          const bar = el("path", { d: barPath(x0, y0, w, bh, 4, true), fill: color(s.slot), class: "mark" }, f.svg);
          if (S.length === 1 && cats.length <= 15)
            el("text", { x: xs(v) + 6, y: y0 + bh / 2 + 4, class: "tick value" }, f.svg).textContent = compact(v);
          bar.addEventListener("mousemove", (e) => tip.show(`<b>${c}</b><div>${swatch(s.slot)}${S.length > 1 ? s.name + ": " : ""}<b>${full(v)}</b></div>`,
            e.clientX - container.getBoundingClientRect().left, e.clientY - container.getBoundingClientRect().top));
          bar.addEventListener("mouseleave", tip.hide);
        });
      });
    }
  }

  function pieChart(container, fig) {
    const vals = fig.series[0].values, cats = fig.categories;
    const total = vals.reduce((a, b) => a + Math.max(0, b), 0) || 1;
    const wrap = document.createElement("div");
    wrap.className = "viz-pie";
    container.appendChild(wrap);
    const size = 220, R = size / 2 - 4, r = R * 0.6, cx = size / 2, cy = size / 2;
    const svg = el("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, class: "viz", role: "img" });
    wrap.appendChild(svg);
    const tip = tooltip(container);
    let a0 = -Math.PI / 2;
    vals.forEach((v, i) => {
      const a1 = a0 + (Math.max(0, v) / total) * Math.PI * 2;
      const large = a1 - a0 > Math.PI ? 1 : 0;
      const p = (a, rad) => [cx + rad * Math.cos(a), cy + rad * Math.sin(a)];
      const [x0, y0] = p(a0, R), [x1, y1] = p(a1 - 1e-6, R), [x2, y2] = p(a1 - 1e-6, r), [x3, y3] = p(a0, r);
      const seg = el("path", { d: `M${x0},${y0}A${R},${R} 0 ${large} 1 ${x1},${y1}L${x2},${y2}A${r},${r} 0 ${large} 0 ${x3},${y3}Z`,
        fill: color(i), stroke: css("--surface"), "stroke-width": 2, class: "mark" }, svg);
      seg.addEventListener("mousemove", (e) => tip.show(`<b>${cats[i]}</b><div>${swatch(i)}<b>${full(v)}</b> · ${(v / total * 100).toFixed(1)}%</div>`,
        e.clientX - container.getBoundingClientRect().left, e.clientY - container.getBoundingClientRect().top));
      seg.addEventListener("mouseleave", tip.hide);
      a0 = a1;
    });
    el("text", { x: cx, y: cy - 2, "text-anchor": "middle", class: "pie-total" }, svg).textContent = compact(total);
    el("text", { x: cx, y: cy + 16, "text-anchor": "middle", class: "tick" }, svg).textContent = "total";
    const list = document.createElement("ul");
    list.className = "pie-legend";
    list.innerHTML = cats.map((c, i) => `<li>${swatch(i)}<span>${c}</span><b>${(vals[i] / total * 100).toFixed(1)}%</b><em>${compact(vals[i])}</em></li>`).join("");
    wrap.appendChild(list);
  }

  function scatterChart(container, fig) {
    legend(container, fig.series);
    const f = frame(container, fig, {});
    const pts = fig.series.flatMap((s) => s.points);
    const xt = niceTicks(Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0])));
    const yt = niceTicks(Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1])));
    const xs = (v) => f.m.l + ((v - xt[0]) / (xt.at(-1) - xt[0])) * f.iw;
    const ys = (v) => f.m.t + f.ih - ((v - yt[0]) / (yt.at(-1) - yt[0])) * f.ih;
    yAxis(f, yt, ys, fig.y_label);
    const g = el("g", {}, f.svg);
    xt.forEach((v) => el("text", { x: xs(v), y: f.H - f.m.b + 18, "text-anchor": "middle", class: "tick" }, g).textContent = compact(v));
    el("text", { x: f.m.l + f.iw / 2, y: f.H - 6, "text-anchor": "middle", class: "axis-label" }, g).textContent = fig.x_label;
    const tip = tooltip(container);
    fig.series.forEach((s) => s.points.forEach(([x, y]) => {
      const c = el("circle", { cx: xs(x), cy: ys(y), r: 4, fill: color(s.slot), "fill-opacity": 0.75,
        stroke: css("--surface"), "stroke-width": 1, class: "mark" }, f.svg);
      c.addEventListener("mousemove", (e) => tip.show(`${fig.series.length > 1 ? `<b>${s.name}</b>` : ""}<div>${fig.x_label}: <b>${full(x)}</b></div><div>${fig.y_label}: <b>${full(y)}</b></div>`,
        xs(x), e.clientY - container.getBoundingClientRect().top));
      c.addEventListener("mouseleave", tip.hide);
    }));
  }

  function render(container, fig) {
    container.innerHTML = "";
    if (!fig || !fig.series || !fig.series.length) { container.textContent = "No data"; return; }
    try {
      if (fig.kind === "line" || fig.kind === "area") lineChart(container, fig);
      else if (fig.kind === "bar" || fig.kind === "hbar" || fig.kind === "histogram") barChart(container, fig);
      else if (fig.kind === "pie") pieChart(container, fig);
      else if (fig.kind === "scatter") scatterChart(container, fig);
      if (fig.note) { const n = document.createElement("p"); n.className = "viz-note"; n.textContent = fig.note; container.appendChild(n); }
    } catch (err) {
      container.textContent = "Could not draw this chart: " + err.message;
    }
  }

  function renderAll() {
    document.querySelectorAll("[data-figure]").forEach((node) => {
      if (node.offsetParent === null) return; // hidden tab; drawn when shown
      render(node, JSON.parse(node.dataset.figure));
    });
  }

  window.Viz = { render, renderAll };
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(renderAll, 150); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
})();
