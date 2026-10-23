// Jádro: obrázek (RGBA) -> 3 vrstvy pro laser. Bez závislostí, běží v prohlížeči i v Node.
(function (root) {
  "use strict";

  // ---------- pomocné masky ----------
  function edt(mask, W, H, wantIdx) {
    // čtvercová euklidovská vzdálenost k nejbližšímu pixelu, kde mask=1 (Felzenszwalb)
    const INF = 1e20, N = W * H;
    const d = new Float64Array(N), idx = wantIdx ? new Int32Array(N) : null;
    const f = new Float64Array(Math.max(W, H)), fi = new Int32Array(Math.max(W, H));
    const v = new Int32Array(Math.max(W, H)), z = new Float64Array(Math.max(W, H) + 1);
    const dd = new Float64Array(Math.max(W, H)), di = new Int32Array(Math.max(W, H));
    function pass(n) {
      let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
      for (let q = 1; q < n; q++) {
        let s;
        while (true) {
          s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
          if (s <= z[k]) { k--; if (k < 0) { k = 0; break; } } else break;
        }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
      }
      k = 0;
      for (let q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        dd[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
        di[q] = fi[v[k]];
      }
    }
    // sloupce
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) { const i = y * W + x; f[y] = mask[i] ? 0 : INF; fi[y] = i; }
      pass(H);
      for (let y = 0; y < H; y++) { const i = y * W + x; d[i] = dd[y]; if (idx) idx[i] = di[y]; }
    }
    // řádky
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) { const i = y * W + x; f[x] = d[i]; fi[x] = idx ? idx[i] : 0; }
      pass(W);
      for (let x = 0; x < W; x++) { const i = y * W + x; d[i] = dd[x]; if (idx) idx[i] = di[x]; }
    }
    return { d, idx };
  }
  const not = (m) => { const o = new Uint8Array(m.length); for (let i = 0; i < m.length; i++) o[i] = m[i] ? 0 : 1; return o; };
  const and = (a, b) => { const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] && b[i] ? 1 : 0; return o; };
  const or = (a, b) => { const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] || b[i] ? 1 : 0; return o; };
  const andNot = (a, b) => { const o = new Uint8Array(a.length); for (let i = 0; i < a.length; i++) o[i] = a[i] && !b[i] ? 1 : 0; return o; };
  const any = (m) => { for (let i = 0; i < m.length; i++) if (m[i]) return true; return false; };

  function dilate(m, W, H, r) {
    if (r <= 0) return m;
    const { d } = edt(m, W, H, false), r2 = r * r, o = new Uint8Array(m.length);
    for (let i = 0; i < m.length; i++) o[i] = d[i] <= r2 ? 1 : 0;
    return o;
  }
  function erode(m, W, H, r) { return r <= 0 ? m : not(dilate(not(m), W, H, r)); }
  const open = (m, W, H, r) => dilate(erode(m, W, H, r), W, H, r);
  const close = (m, W, H, r) => erode(dilate(m, W, H, r), W, H, r);

  function smooth(m, W, H, sigma) {
    const b = blurF(m, W, H, sigma), o = new Uint8Array(m.length);
    for (let i = 0; i < m.length; i++) o[i] = b[i] > 0.5 ? 1 : 0;
    return o;
  }
  function blurF(m, W, H, sigma) {
    const rad = Math.ceil(sigma * 3), k = [];
    let s = 0;
    for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); s += v; }
    for (let i = 0; i < k.length; i++) k[i] /= s;
    const a = new Float32Array(m.length), b = new Float32Array(m.length);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let j = -rad; j <= rad; j++) { const xx = Math.min(W - 1, Math.max(0, x + j)); acc += m[y * W + xx] * k[j + rad]; }
      a[y * W + x] = acc;
    }
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let j = -rad; j <= rad; j++) { const yy = Math.min(H - 1, Math.max(0, y + j)); acc += a[yy * W + x] * k[j + rad]; }
      b[y * W + x] = acc;
    }
    return b;
  }

  // Obrysy s přesností pod pixel: maska se lehce rozmaže a hranice se hledá interpolací na úrovni 0,5.
  // Výsledek nemá schody po pixelech (jako trasování v grafických programech).
  function contoursSub(m, W, H, sigma) {
    const f = blurF(m, W, H, sigma || 0.8);
    const PW = W + 2;
    const v = (x, y) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : f[y * W + x]);
    const hid = (x, y) => ((y + 1) * PW + (x + 1)) * 2, vid = (x, y) => ((y + 1) * PW + (x + 1)) * 2 + 1;
    const pt = (id) => {
      const cell = id >> 1, x = (cell % PW) - 1, y = Math.floor(cell / PW) - 1;
      if ((id & 1) === 0) { const a = v(x, y), b = v(x + 1, y), t = b !== a ? (0.5 - a) / (b - a) : 0.5; return [x + t, y]; }
      const a = v(x, y), b = v(x, y + 1), t = b !== a ? (0.5 - a) / (b - a) : 0.5; return [x, y + t];
    };
    const next = new Map();
    for (let y = -1; y < H; y++) for (let x = -1; x < W; x++) {
      const tl = v(x, y) > 0.5 ? 1 : 0, tr = v(x + 1, y) > 0.5 ? 1 : 0, br = v(x + 1, y + 1) > 0.5 ? 1 : 0, bl = v(x, y + 1) > 0.5 ? 1 : 0;
      const c = tl * 8 + tr * 4 + br * 2 + bl;
      if (c === 0 || c === 15) continue;
      const T = hid(x, y), R = vid(x + 1, y), B = hid(x, y + 1), L = vid(x, y);
      const add = (a, b) => next.set(a, b);
      switch (c) {
        case 1: add(L, B); break; case 2: add(B, R); break; case 3: add(L, R); break; case 4: add(R, T); break;
        case 5: add(L, T); add(R, B); break; case 6: add(B, T); break; case 7: add(L, T); break; case 8: add(T, L); break;
        case 9: add(T, B); break; case 10: add(T, R); add(B, L); break; case 11: add(T, R); break; case 12: add(R, L); break;
        case 13: add(R, B); break; case 14: add(B, L); break;
      }
    }
    const polys = [];
    for (const start of next.keys()) {
      if (!next.has(start)) continue;
      const ids = [start];
      let k = next.get(start);
      next.delete(start);
      while (k !== undefined && k !== start && next.has(k)) { ids.push(k); const n2 = next.get(k); next.delete(k); k = n2; }
      if (ids.length >= 4) polys.push(ids.map(pt));
    }
    return polys;
  }

  function label(m, W, H) {
    // 8-souvislé komponenty
    const lab = new Int32Array(m.length), sizes = [0];
    const stack = new Int32Array(m.length);
    let n = 0;
    for (let s = 0; s < m.length; s++) {
      if (!m[s] || lab[s]) continue;
      n++; let sp = 0, cnt = 0; stack[sp++] = s; lab[s] = n;
      while (sp) {
        const i = stack[--sp]; cnt++;
        const x = i % W, y = (i / W) | 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          const j = yy * W + xx;
          if (m[j] && !lab[j]) { lab[j] = n; stack[sp++] = j; }
        }
      }
      sizes.push(cnt);
    }
    return { lab, n, sizes };
  }

  // ---------- barvy ----------
  function kmeans(px, k, iters) {
    // px: Float32Array [r,g,b,...]
    const n = px.length / 3;
    let rnd = 12345;
    const rand = () => { rnd = (rnd * 1103515245 + 12345) & 0x7fffffff; return rnd / 0x7fffffff; };
    // k-means++
    const c = [];
    let i0 = Math.floor(rand() * n);
    c.push([px[i0 * 3], px[i0 * 3 + 1], px[i0 * 3 + 2]]);
    const dist = new Float64Array(n).fill(Infinity);
    while (c.length < k) {
      const last = c[c.length - 1];
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const dr = px[i * 3] - last[0], dg = px[i * 3 + 1] - last[1], db = px[i * 3 + 2] - last[2];
        const d = dr * dr + dg * dg + db * db;
        if (d < dist[i]) dist[i] = d;
        sum += dist[i];
      }
      let t = rand() * sum, j = 0;
      for (; j < n - 1; j++) { t -= dist[j]; if (t <= 0) break; }
      c.push([px[j * 3], px[j * 3 + 1], px[j * 3 + 2]]);
    }
    const as = new Int32Array(n);
    for (let it = 0; it < iters; it++) {
      const s = c.map(() => [0, 0, 0, 0]);
      for (let i = 0; i < n; i++) {
        let best = 0, bd = Infinity;
        for (let j = 0; j < k; j++) {
          const dr = px[i * 3] - c[j][0], dg = px[i * 3 + 1] - c[j][1], db = px[i * 3 + 2] - c[j][2];
          const d = dr * dr + dg * dg + db * db;
          if (d < bd) { bd = d; best = j; }
        }
        as[i] = best;
        s[best][0] += px[i * 3]; s[best][1] += px[i * 3 + 1]; s[best][2] += px[i * 3 + 2]; s[best][3]++;
      }
      for (let j = 0; j < k; j++) if (s[j][3]) c[j] = [s[j][0] / s[j][3], s[j][1] / s[j][3], s[j][2] / s[j][3]];
    }
    return c;
  }
  function flatSample(get, fg, W, H, maxN) {
    const N = W * H, flat = new Uint8Array(N);
    let nf = 0, nfg = 0;
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!fg[i]) continue;
      nfg++;
      let ok = 1;
      for (const j of [i - 1, i + 1, i - W, i + W]) {
        if (!fg[j]) { ok = 0; break; }
        const d = Math.abs(get(i, 0) - get(j, 0)) + Math.abs(get(i, 1) - get(j, 1)) + Math.abs(get(i, 2) - get(j, 2));
        if (d > 24) { ok = 0; break; }
      }
      if (ok) { flat[i] = 1; nf++; }
    }
    const use = nf > nfg * 0.2 ? flat : fg, cnt = nf > nfg * 0.2 ? nf : nfg;
    const step = Math.max(1, Math.floor(cnt / maxN)), out = [];
    let j = 0;
    for (let i = 0; i < N; i++) if (use[i] && j++ % step === 0) out.push(get(i, 0), get(i, 1), get(i, 2));
    return new Float32Array(out);
  }
  // téměř stejné barvy sloučit (k-means někdy rozdělí jednu plochu na dva odstíny)
  function dedupe(cent, thr) {
    const out = [];
    for (const c of cent) if (!out.some((o) => Math.hypot(o[0] - c[0], o[1] - c[1], o[2] - c[2]) < thr)) out.push(c);
    return out;
  }
  const lum = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  function toLab(c) {   // sRGB -> CIE Lab (D65)
    const f = (v) => { v /= 255; return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    const r = f(c[0]), g = f(c[1]), b = f(c[2]);
    const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047, Y = 0.2126 * r + 0.7152 * g + 0.0722 * b, Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    const t = (v) => (v > 0.008856 ? Math.cbrt(v) : 7.787 * v + 16 / 116);
    return [116 * t(Y) - 16, 500 * (t(X) - t(Y)), 200 * (t(Y) - t(Z))];
  }

  // Pozadí: průhlednost, nebo (u JPG) bílé/jednolité pozadí zaplavené od okraje
  // Pozadí: průhlednost, nebo barva rohů zaplavená od okraje. bg.tol = citlivost,
  // bg.bgSeeds = kliknutá místa pozadí, bg.fgSeeds = kliknutá místa postavy (indexy pixelů)
  function foreground(rgba, W, H, bg) {
    bg = bg || {};
    const N = W * H, fg = new Uint8Array(N), tol = bg.tol == null ? 30 : bg.tol, tol2 = tol * tol;
    const col = (i) => [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
    const near = (i, c) => { const dr = rgba[i * 4] - c[0], dg = rgba[i * 4 + 1] - c[1], db = rgba[i * 4 + 2] - c[2]; return dr * dr + dg * dg + db * db < tol2; };
    const flood = (seeds, ok, mark) => {
      const stack = seeds.slice();
      while (stack.length) {
        const i = stack.pop();
        if (mark[i] || !ok(i)) continue;
        mark[i] = 1;
        const x = i % W, y = (i / W) | 0;
        if (x > 0) stack.push(i - 1);
        if (x < W - 1) stack.push(i + 1);
        if (y > 0) stack.push(i - W);
        if (y < H - 1) stack.push(i + W);
      }
    };
    const transparentPx = (i) => rgba[i * 4 + 3] < 128;
    let transparent = 0;
    for (let i = 0; i < N; i++) if (transparentPx(i)) transparent++;
    const back = new Uint8Array(N);
    const gapR0 = Math.max(2, Math.round(Math.max(W, H) / 400));
    // pozadí = místa pozadí dosažitelná od okraje; neprotéká mezerami v obrysu (bariéra = postava rozšířená o pár pixelů)
    const floodOutside = (isBg, seeds, gr) => {
      const gapR = gr || gapR0;
      const notBg = new Uint8Array(N);
      for (let i = 0; i < N; i++) notBg[i] = isBg(i) ? 0 : 1;
      const barrier = dilate(notBg, W, H, gapR);
      flood(seeds.filter((i) => !barrier[i]), (i) => isBg(i) && !barrier[i], back);
      // dorovnat pozadí až k postavě (pás, který zabrala bariéra), ale jen tam, kam se dosáhne zvenku
      const d = edt(back, W, H, false).d, lim = (gapR + 1) * (gapR + 1);
      for (let i = 0; i < N; i++) if (!back[i] && isBg(i) && d[i] <= lim) back[i] = 1;
    };
    if (transparent > N * 0.01) {
      // kresba jen z tahů na průhledném pozadí (omalovánka v PNG): vnitřek uzavřených tvarů patří k postavě (bílá plocha)
      let opaque = 0, dark = 0;
      for (let i = 0; i < N; i++) if (!transparentPx(i)) { opaque++; if (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2] < 110) dark++; }
      if (opaque > 0 && dark > opaque * 0.8) {
        const border = [];
        for (let x = 0; x < W; x++) border.push(x, (H - 1) * W + x);
        for (let y = 0; y < H; y++) border.push(y * W, y * W + W - 1);
        // náčrtové tahy mívají mezery (vlasy Ricka): bariéra širší než u barevných obrázků
        floodOutside(transparentPx, border.filter((i) => transparentPx(i)), Math.max(gapR0, Math.round(Math.max(W, H) / 150)));
      } else for (let i = 0; i < N; i++) back[i] = transparentPx(i) ? 1 : 0;
    } else {
      // barva pozadí = nejčastější barva v pásu u okraje (ignoruje tenký rámeček obrázku)
      const band = Math.max(3, Math.round(Math.min(W, H) * 0.012)), cnt = new Map();
      const bandPx = [];
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        if (x >= band && y >= band && x < W - band && y < H - band) continue;
        const i = y * W + x; bandPx.push(i);
        const key = (rgba[i * 4] >> 3) * 1024 + (rgba[i * 4 + 1] >> 3) * 32 + (rgba[i * 4 + 2] >> 3);
        cnt.set(key, (cnt.get(key) || 0) + 1);
      }
      let bestK = 0, bestN = -1;
      for (const [k2, n2] of cnt) if (n2 > bestN) { bestN = n2; bestK = k2; }
      const ref = [((bestK >> 10) & 31) * 8 + 4, ((bestK >> 5) & 31) * 8 + 4, (bestK & 31) * 8 + 4];
      const seeds = bandPx.filter((i) => near(i, ref));
      // Pozadí se nesmí protéct do postavy malými mezerami v obrysu (bílá rukavice na bílém pozadí).
      floodOutside((i) => transparentPx(i) || near(i, ref), seeds);
      // tenký rámeček obrázku u okraje (jiná barva než pozadí) taky patří k pozadí
      for (const i of bandPx) back[i] = 1;
      // nakreslený rámeček stránky (omalovánka): rovné čáry po všech čtyřech stranách postavy -> pozadí, pak znovu zaplavit
      let x0 = W, x1 = -1, y0 = H, y1 = -1;
      const C0 = label(not(back), W, H);   // obdélník největšího kusu (číslo stránky pod rámečkem nepočítat)
      let q0 = 0; for (let q = 1; q <= C0.n; q++) if (!q0 || C0.sizes[q] > C0.sizes[q0]) q0 = q;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (q0 && C0.lab[y * W + x] === q0) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (x1 > x0 + 20 && y1 > y0 + 20) {
        const bw = x1 - x0 + 1, bh = y1 - y0 + 1, t = Math.max(3, Math.round(0.03 * Math.min(bw, bh)));
        const rowN = (y) => { let c = 0; for (let x = x0; x <= x1; x++) c += back[y * W + x] ? 0 : 1; return c; };
        const colN = (x) => { let c = 0; for (let y = y0; y <= y1; y++) c += back[y * W + x] ? 0 : 1; return c; };
        const rows = [], cols = [];
        for (let y = y0; y < y0 + t; y++) if (rowN(y) > 0.6 * bw) rows.push(y);
        for (let y = y1; y > y1 - t; y--) if (rowN(y) > 0.6 * bw) rows.push(y);
        for (let x = x0; x < x0 + t; x++) if (colN(x) > 0.6 * bh) cols.push(x);
        for (let x = x1; x > x1 - t; x--) if (colN(x) > 0.6 * bh) cols.push(x);
        const has = (arr, lo, hi) => arr.some((v) => v >= lo && v <= hi);
        if (has(rows, y0, y0 + t) && has(rows, y1 - t, y1) && has(cols, x0, x0 + t) && has(cols, x1 - t, x1)) {
          const frame = new Uint8Array(N);
          for (const y of rows) for (let x = x0; x <= x1; x++) for (let d = -1; d <= 1; d++) { const yy = y + d; if (yy >= 0 && yy < H) frame[yy * W + x] = 1; }
          for (const x of cols) for (let y = y0; y <= y1; y++) for (let d = -1; d <= 1; d++) { const xx = x + d; if (xx >= 0 && xx < W) frame[y * W + xx] = 1; }
          back.fill(0);
          floodOutside((i) => transparentPx(i) || near(i, ref) || frame[i], bandPx.filter((i) => near(i, ref) || frame[i]));
          for (let i = 0; i < N; i++) if (frame[i]) back[i] = 1;
          for (const i of bandPx) back[i] = 1;
          // zbytky rámečku (zaoblené rohy) a jiné drobky mimo hlavní kresbu pryč
          const fgm = not(back), C = label(fgm, W, H);
          let big = 0; for (let q = 1; q <= C.n; q++) if (C.sizes[q] > big) big = C.sizes[q];
          for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < big * 0.02) back[i] = 1;
        }
      }
    }
    // kliknutá místa pozadí: zaplavit podobnou barvu od kliku
    for (const s0 of bg.bgSeeds || []) { const c = col(s0), m = new Uint8Array(N); flood([s0], (i) => !transparentPx(i) && near(i, c), m); for (let i = 0; i < N; i++) if (m[i]) back[i] = 1; }
    // kliknutá místa postavy: vrátit podobnou barvu, která skončila v pozadí
    for (const s0 of bg.fgSeeds || []) {
      if (transparentPx(s0)) {
        // klik do průhledného místa kresby (vlasy Ricka, kam proteklo pozadí mezerami v tazích): uzavřít mezery
        // čím dál širší bariérou, dokud oblast od kliku nepřestane sahat k okraji obrázku
        const opaque = new Uint8Array(N);
        for (let i = 0; i < N; i++) opaque[i] = transparentPx(i) ? 0 : 1;
        for (const k of [1, 2, 3, 5, 8]) {
          const g = Math.max(2, Math.round(Math.max(W, H) / 150)) * k, bar = dilate(opaque, W, H, g);
          if (bar[s0]) continue;
          const m = new Uint8Array(N);
          flood([s0], (i) => !bar[i], m);
          let edge = false;
          for (let x = 0; x < W && !edge; x++) edge = m[x] || m[(H - 1) * W + x];
          for (let y = 0; y < H && !edge; y++) edge = m[y * W] || m[y * W + W - 1];
          if (edge) continue;
          const grow = dilate(m, W, H, g + 1);
          for (let i = 0; i < N; i++) if (grow[i] && transparentPx(i)) back[i] = 0;
          break;
        }
        continue;
      }
      const c = col(s0), m = new Uint8Array(N); flood([s0], (i) => back[i] && !transparentPx(i) && near(i, c), m); for (let i = 0; i < N; i++) if (m[i]) back[i] = 0;
    }
    for (let i = 0; i < N; i++) fg[i] = back[i] ? 0 : 1;
    return fg;
  }

  // ---------- kontury (marching squares) ----------
  function contours(m, W, H, eps) {
    // vrací uzavřené polygony v pixelech, hrana na rozhraní pixelů
    const PW = W + 2, PH = H + 2;
    const g = (x, y) => (x >= 1 && y >= 1 && x <= W && y <= H ? m[(y - 1) * W + (x - 1)] : 0);
    const segs = new Map(); // start key -> end point
    const key = (x, y) => x * 2 + 1 + (y * 2 + 1) * (PW * 2 + 4);
    const add = (ax, ay, bx, by) => { segs.set(key(ax, ay), [bx, by, ax, ay]); };
    for (let y = 0; y < PH - 1; y++) for (let x = 0; x < PW - 1; x++) {
      const tl = g(x, y), tr = g(x + 1, y), br = g(x + 1, y + 1), bl = g(x, y + 1);
      const c = tl * 8 + tr * 4 + br * 2 + bl;
      if (c === 0 || c === 15) continue;
      // hraniční body (souřadnice hran buňky, pixel středy v x, x+1)
      const T = [x + 0.5, y], R = [x + 1, y + 0.5], B = [x + 0.5, y + 1], L = [x, y + 0.5];
      // orientace: materiál po levé ruce
      switch (c) {
        case 1: add(...L, ...B); break;
        case 2: add(...B, ...R); break;
        case 3: add(...L, ...R); break;
        case 4: add(...R, ...T); break;
        case 5: add(...L, ...T); add(...R, ...B); break;
        case 6: add(...B, ...T); break;
        case 7: add(...L, ...T); break;
        case 8: add(...T, ...L); break;
        case 9: add(...T, ...B); break;
        case 10: add(...T, ...R); add(...B, ...L); break;
        case 11: add(...T, ...R); break;
        case 12: add(...R, ...L); break;
        case 13: add(...R, ...B); break;
        case 14: add(...B, ...L); break;
      }
    }
    const polys = [];
    for (const [k0, s0] of segs) {
      if (!segs.has(k0)) continue;
      const pts = [[s0[2], s0[3]]];
      let k = k0;
      while (segs.has(k)) {
        const s = segs.get(k);
        segs.delete(k);
        pts.push([s[0], s[1]]);
        k = key(s[0], s[1]);
      }
      if (pts.length > 3) { const q = pts.map((p) => [p[0] - 0.5, p[1] - 0.5]); polys.push(eps < 0 ? q : rdp(q, eps)); }
    }
    return polys.filter((p) => p.length >= 3);
  }
  function rdp(pts, eps) {
    // Ramer-Douglas-Peucker na uzavřené křivce
    if (pts.length < 4) return pts;
    const keep = new Uint8Array(pts.length);
    // rozdělit křivku v nejvzdálenějším bodě od startu
    let far = 0, fd = 0;
    for (let i = 1; i < pts.length; i++) { const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2; if (d > fd) { fd = d; far = i; } }
    keep[0] = keep[far] = keep[pts.length - 1] = 1;
    const stack = [[0, far], [far, pts.length - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      const [ax, ay] = pts[a], [bx, by] = pts[b];
      const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
      let mi = -1, md = 0;
      for (let i = a + 1; i < b; i++) {
        const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + bx * ay - by * ax) / len;
        if (d > md) { md = d; mi = i; }
      }
      if (mi >= 0 && md > eps) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
    }
    const out = [];
    for (let i = 0; i < pts.length - 1; i++) if (keep[i]) out.push(pts[i]);
    return out;
  }

  function rdpOpen(pts, eps) {
    if (pts.length < 3) return pts;
    const [fx, fy] = pts[0], [lx, ly] = pts[pts.length - 1];
    if (Math.hypot(fx - lx, fy - ly) < 1e-6) { const c = rdp(pts, eps); return c.length >= 3 ? c.concat([c[0]]) : c; }
    const keep = new Uint8Array(pts.length); keep[0] = keep[pts.length - 1] = 1;
    const stack = [[0, pts.length - 1]];
    while (stack.length) {
      const [a, b] = stack.pop();
      const [ax, ay] = pts[a], [bx, by] = pts[b];
      const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
      let mi = -1, md = 0;
      for (let i = a + 1; i < b; i++) { const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + bx * ay - by * ax) / len; if (d > md) { md = d; mi = i; } }
      if (mi >= 0 && md > eps) { keep[mi] = 1; stack.push([a, mi], [mi, b]); }
    }
    return pts.filter((_, i) => keep[i]);
  }
  // čáry nátěru: v SVG otevřené cesty v barvě nátěru
  let curveSmooth = 0;   // 0 = lomené čáry, 1..5 = hladké křivky (síla vyhlazení)
  function closedD(p, mm, ox, oy) {
    if (curveSmooth > 0) { const d = smoothPath(p, 0.15 + 0.08 * curveSmooth, (60 * Math.PI) / 180, curveSmooth, mm, ox, oy); if (d) return d; }
    return "M" + p.map((q) => (q[0] * mm + ox).toFixed(3) + "," + (q[1] * mm + oy).toFixed(3)).join(" L") + "z";
  }
  function openD(l, mm, ox, oy) {
    let p = l;
    if (curveSmooth > 0 && p.length > 4) {
      for (let pass = 0; pass < curveSmooth; pass++) p = p.map((q, i) => (i === 0 || i === p.length - 1 ? q : [(p[i - 1][0] + 2 * q[0] + p[i + 1][0]) / 4, (p[i - 1][1] + 2 * q[1] + p[i + 1][1]) / 4]));
      p = rdpOpen(p, 0.4 + 0.15 * curveSmooth);
    }
    return "M" + p.map((q) => (q[0] * mm + ox).toFixed(3) + "," + (q[1] * mm + oy).toFixed(3)).join(" L");
  }
  const paintPaths = (paint, mm, ox, oy) => paint.map((pp) => `<path style="stroke:${pp.color};stroke-width:0.05;fill:none" d="${pp.lines.map((l) => openD(l, mm, ox, oy)).join(" ")}"/>\n`).join("");
  const ENGRAVE = "#ff00ff";
  const engravePath = (polys, mm, ox, oy) => polys.length ? `<path style="stroke:${ENGRAVE};stroke-width:0.05;fill:${ENGRAVE};fill-rule:evenodd" d="${polys.map((p) => closedD(p, mm, ox, oy)).join(" ")}"/>\n` : "";
  function svg(wmm, hmm, mm, cut, mark, title, paint, engr) {
    const d = (polys) => polys.map((p) => closedD(p, mm, 0, 0)).join(" ");
    let s = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${wmm.toFixed(2)}mm" height="${hmm.toFixed(2)}mm" viewBox="0 0 ${wmm.toFixed(3)} ${hmm.toFixed(3)}"><title>${title}</title>\n`;
    if (cut.length) s += `<path style="stroke:#000000;stroke-width:0.05;fill:none" d="${d(cut)}"/>\n`;
    if (mark.length) s += `<path style="stroke:#00e0e0;stroke-width:0.05;fill:none" d="${d(mark)}"/>\n`;
    if (paint && paint.length) s += paintPaths(paint, mm, 0, 0);
    if (engr && engr.length) s += engravePath(engr, mm, 0, 0);
    return s + "</svg>\n";
  }
  // všechny listy do jednoho SVG: mřížka po dvou, nad každým listem červený popisek
  function combined(wmm, hmm, mm, sheets) {
    const d = (polys, ox, oy) => polys.map((p) => closedD(p, mm, ox, oy)).join(" ");
    const gap = 10, lab = 14, cols = 2, rows = Math.ceil(sheets.length / cols);
    const W = cols * wmm + (cols - 1) * gap, H = rows * (hmm + lab) + (rows - 1) * gap;
    let s = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${W.toFixed(2)}mm" height="${H.toFixed(2)}mm" viewBox="0 0 ${W.toFixed(3)} ${H.toFixed(3)}"><title>vrstvy pro laser</title>\n`;
    sheets.forEach((sh, i) => {
      const ox = (i % cols) * (wmm + gap), oy = Math.floor(i / cols) * (hmm + lab + gap);
      s += `<g id="vrstva${i + 1}">\n<text x="${(ox + wmm / 2).toFixed(2)}" y="${(oy + lab * 0.72).toFixed(2)}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff0000">${sh.label}</text>\n`;
      if (sh.cut.length) s += `<path style="stroke:#000000;stroke-width:0.05;fill:none" d="${d(sh.cut, ox, oy + lab)}"/>\n`;
      if (sh.mark.length) s += `<path style="stroke:#00e0e0;stroke-width:0.05;fill:none" d="${d(sh.mark, ox, oy + lab)}"/>\n`;
      if (sh.paint && sh.paint.length) s += paintPaths(sh.paint, mm, ox, oy + lab);
      if (sh.engrave && sh.engrave.length) s += engravePath(sh.engrave, mm, ox, oy + lab);
      s += "</g>\n";
    });
    return s + "</svg>\n";
  }

  // ---------- 1. krok: barvy ----------
  function analyze(rgba0, W, H, nColors, bg) {
    const fg = foreground(rgba0, W, H, bg);
    // barvy z obrázku položeného na bílý papír (průhledné a poloprůhledné pixely postavy = bílá)
    const rgba = new Uint8ClampedArray(rgba0.length);
    for (let i = 0; i < W * H; i++) {
      const a = rgba0[i * 4 + 3] / 255;
      for (let ch = 0; ch < 3; ch++) rgba[i * 4 + ch] = rgba0[i * 4 + ch] * a + 255 * (1 - a);
      rgba[i * 4 + 3] = 255;
    }
    let cnt = 0;
    for (let i = 0; i < fg.length; i++) if (fg[i]) cnt++;
    let cent = dedupe(kmeans(flatSample((i, ch) => rgba[i * 4 + ch], fg, W, H, 40000), nColors, 12), 22);
    cent.sort((a, b) => lum(a) - lum(b));
    const lab = new Int8Array(W * H).fill(-1);
    for (let i = 0; i < fg.length; i++) {
      if (!fg[i]) continue;
      let best = 0, bd = Infinity;
      for (let c = 0; c < cent.length; c++) {
        const dr = rgba[i * 4] - cent[c][0], dg = rgba[i * 4 + 1] - cent[c][1], db = rgba[i * 4 + 2] - cent[c][2];
        const d = dr * dr + dg * dg + db * db;
        if (d < bd) { bd = d; best = c; }
      }
      lab[i] = best;
    }
    // přechodové barvy (jen tenké lemy kolem linek) a malé plochy přebarvit na nejbližší větší
    const minPx = Math.max(16, Math.round(W * H * 0.00006));
    const thinR = Math.max(2, Math.round(Math.max(W, H) / 400));
    const bad = new Uint8Array(W * H);
    for (let c = 0; c < cent.length; c++) {
      const m = new Uint8Array(W * H);
      let area = 0;
      for (let i = 0; i < m.length; i++) if (lab[i] === c) { m[i] = 1; area++; }
      if (!area) continue;
      if (c > 0) {
        const core = open(m, W, H, thinR);
        let ca = 0;
        for (let i = 0; i < core.length; i++) ca += core[i];
        if (ca < area * 0.25) { for (let i = 0; i < m.length; i++) if (m[i]) bad[i] = 1; continue; }
      }
      const L = label(m, W, H);
      for (let i = 0; i < m.length; i++) if (L.lab[i] && L.sizes[L.lab[i]] < minPx) bad[i] = 1;
    }
    const good = new Uint8Array(W * H);
    for (let i = 0; i < good.length; i++) good[i] = lab[i] >= 0 && !bad[i] ? 1 : 0;
    if (any(bad)) {
      const { idx } = edt(good, W, H, true);
      for (let i = 0; i < bad.length; i++) if (bad[i]) lab[i] = lab[idx[i]];
    }
    // vyčistit hrany barev: přechodové pixely připadnou většinové sousední barvě
    for (let it = 0; it < 2; it++) {
      const out = Int8Array.from(lab), cnt = new Int16Array(cent.length);
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        if (lab[i] < 0) continue;
        cnt.fill(0);
        let best = lab[i], bc = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const v = lab[i + dy * W + dx];
          if (v >= 0 && ++cnt[v] > bc) { bc = cnt[v]; best = v; }
        }
        if (bc >= 5 && cnt[lab[i]] <= 2) out[i] = best;
      }
      lab.set(out);
    }
    // vyřadit prázdné barvy, spočítat plochy
    const area = cent.map(() => 0);
    for (let i = 0; i < lab.length; i++) if (lab[i] >= 0) area[lab[i]]++;
    // skoro stejné odstíny (tmavší lem, stín) sloučit do větší plochy: dvě modré -> jedna
    // Vzdálenost barev jako ji vidí oko (Lab): světle šedá a bílá, tři odstíny krémové u Jerryho -> jedna barva.
    // Drobné plochy (pod 1,5 %) se slučují ochotněji, ať je barev co nejméně.
    const labC = cent.map(toLab), totA = area.reduce((x, y) => x + y, 0) || 1;
    for (;;) {
      let pa = -1, pb = -1, pd = 1e9;
      for (let a = 0; a < cent.length; a++) for (let b = 0; b < cent.length; b++) {
        if (a === b || !area[a] || !area[b] || area[a] > area[b] * 0.6) continue;
        if (lum(cent[a]) < 60 && lum(cent[b]) < 60) continue;          // černé linky nechat
        const d = Math.hypot(labC[a][0] - labC[b][0], labC[a][1] - labC[b][1], labC[a][2] - labC[b][2]);
        if (d < (area[a] < totA * 0.015 ? 24 : 17) && d < pd) { pd = d; pa = a; pb = b; }
      }
      if (pa < 0) break;
      for (let i = 0; i < lab.length; i++) if (lab[i] === pa) lab[i] = pb;
      area[pb] += area[pa]; area[pa] = 0;
    }
    const used = cent.map((_, c) => c).filter((c) => area[c] > 0);
    const remap = new Int8Array(cent.length).fill(-1);
    used.forEach((c, k) => (remap[c] = k));
    for (let i = 0; i < lab.length; i++) if (lab[i] >= 0) lab[i] = remap[lab[i]];
    cent = used.map((c) => cent[c].map(Math.round));
    const areas = used.map((c) => area[c]);
    // výchozí role: tmavé = obrys, bílé/krémové = dřevo, zbytek = nátěr
    const roles = cent.map((c, k) => (k === 0 || lum(c) < 60 ? "outline" : lum(c) > 205 && c[2] <= c[1] + 10 && Math.max(...c) - Math.min(...c) < 70 ? "wood" : 2));
    // další tmavá barva (tmavě hnědá srst, stín) tvořící hlavně plochy, ne tahy, je natřená plocha, ne obrys
    for (let k = 1; k < cent.length; k++) {
      if (roles[k] !== "outline") continue;
      const mk = new Uint8Array(W * H);
      let nk = 0;
      for (let i = 0; i < mk.length; i++) if (lab[i] === k) { mk[i] = 1; nk++; }
      const op = open(mk, W, H, Math.max(3, Math.round(Math.max(W, H) / 200)));
      let nc = 0;
      for (let i = 0; i < op.length; i++) nc += op[i];
      if (nk && nc / nk > 0.5) roles[k] = 2;
    }
    // bílé a krémové plochy: ve výchozím stavu natřená vrstva (bíle nebo holé dřevo), ne součást horní vrstvy
    // nejtmavší barva tvořící hlavně velké plochy (tělo Krtka), ne tenké tahy -> natřená vrstva, ne obrys
    {
      const m0 = new Uint8Array(W * H);
      let n0 = 0;
      for (let i = 0; i < m0.length; i++) if (lab[i] === 0) { m0[i] = 1; n0++; }
      if (n0) {
        const op = open(m0, W, H, Math.max(3, Math.round(Math.max(W, H) / 200)));
        let nc = 0;
        for (let i = 0; i < op.length; i++) nc += op[i];
        // velké černé plochy bez obrysové kresby (auto s černými koly, 3D postavička): díra až na černou desku.
        // Když je ale černou obtažená celá postava (Krtek, Mickey), jde o kresbu: tahy budou dřevěné linky
        // a velké černé plochy z nich nástroj Linky sám udělá dírou (viz lineArt autoHole).
        let nb = 0, nd = 0;
        for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
          const i = y * W + x;
          if (lab[i] < 0 || (lab[i - 1] >= 0 && lab[i + 1] >= 0 && lab[i - W] >= 0 && lab[i + W] >= 0)) continue;
          nb++; if (lab[i] === 0) nd++;
        }
        if (nc / n0 > 0.5 && nd < 0.5 * nb) roles[0] = "board";
      }
    }
    const whiteish = cent.map((c, k) => roles[k] === "wood");
    for (let k = 0; k < roles.length; k++) if (roles[k] === "wood") roles[k] = 2;
    return { W, H, lab, cent, areas, roles, fg, whiteish };
  }

  // ---------- 2. krok: vrstvy ----------
  function build(an, roles, opt) {
    let { W, H, lab, cent } = an;
    // ořez na motiv
    let x0 = W, y0 = H, x1 = -1, y1 = -1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (lab[y * W + x] >= 0) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const mm = opt.width / (x1 - x0 + 1);
    const px = (v) => v / mm;
    const pad = Math.max(0, Math.round(px(opt.frame + opt.gap)));   // záporná mezera = postava zasahuje do rámečku
    const w = x1 - x0 + 1 + 2 * pad, h = y1 - y0 + 1 + 2 * pad, N = w * h;
    const L = new Int8Array(N).fill(-1);
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) L[(y - y0 + pad) * w + (x - x0 + pad)] = lab[y * W + x];
    const cropM = (src) => { const o = new Uint8Array(N); for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) o[(y - y0 + pad) * w + (x - x0 + pad)] = src[y * W + x]; return o; };
    const LD = opt.lineData ? { line: cropM(opt.lineData.line), hole: cropM(opt.lineData.hole) } : null;
    // přesná kopie tahů jen u obrázků, které čáry mají; obrázek bez čar (Krtek) jde osvědčenou cestou
    const EX = !!LD && any(LD.line);

    const isRole = (r) => { const m = new Uint8Array(N); for (let i = 0; i < N; i++) m[i] = L[i] >= 0 && roles[L[i]] === r ? 1 : 0; return m; };
    curveSmooth = Math.max(0, Math.min(5, opt.smooth == null ? 3 : opt.smooth));
    const sig = Math.max(1, px(opt.maskSmooth == null ? 0.25 : opt.maskSmooth)), eps = curveSmooth > 0 ? -1 : Math.max(0.6, px(0.05));
    const r = Math.max(1, Math.round(px(opt.minFeature) / 2));

    // deska, rámeček
    const ch = Math.round(px(opt.chamfer)), f = Math.round(px(opt.frame));
    const rect = new Uint8Array(N), ring = new Uint8Array(N);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const cut = x + y < ch || (w - 1 - x) + y < ch || x + (h - 1 - y) < ch || (w - 1 - x) + (h - 1 - y) < ch;
      rect[i] = cut ? 0 : 1;
      ring[i] = rect[i] && (x < f || y < f || x >= w - f || y >= h - f) ? 1 : 0;
    }

    // postava
    const fgm = new Uint8Array(N);
    for (let i = 0; i < N; i++) fgm[i] = L[i] >= 0 ? 1 : 0;
    let sil = close(open(smooth(fgm, w, h, sig), w, h, r), w, h, r);
    const dropSmall = (m) => {
      const C = label(m, w, h);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] * mm * mm < opt.minPiece) m[i] = 0;
      return m;
    };
    sil = dropSmall(sil);
    if (EX) sil = or(sil, LD.line);
    // počet vrstev a úroveň každé barvy: obrys a dřevo = horní vrstva, nátěr = vrstva 2..NL-1
    const NL = Math.max(3, Math.min(5, opt.layers || 3));
    const lvlOf = roles.map((ro) => (ro === "board" ? 1 : ro === "outline" || ro === "wood" ? NL : Math.max(2, Math.min(NL - 1, ro === "paint" ? 2 : +ro))));
    const lev = new Int8Array(N);
    for (let i = 0; i < N; i++) lev[i] = L[i] >= 0 ? lvlOf[L[i]] : 0;
    const levAtLeast = (k) => { const m = new Uint8Array(N); for (let i = 0; i < N; i++) m[i] = lev[i] >= k ? 1 : 0; return m; };

    const blk = LD ? and(LD.line, sil) : and(smooth(isRole("outline"), w, h, sig), sil);
    // černé plochy skrz na desku: rozhodnout o celém tvaru, ne bod po bodu.
    // Jádro = místa silnější než "Černá skrz od"; platí jen jádra výrazně větší než jeden kruh (zornice, čumák, obočí),
    // ne spoje a ztluštění obrysových tahů. Jádro se pak rozšíří na celý souvislý silný tvar.
    const R = Math.max(r + 1, Math.round(px(opt.hole) / 2));
    let core = LD ? new Uint8Array(N) : open(blk, w, h, R);
    { const C = label(core, w, h), minA = 3 * Math.PI * R * R;
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < minA) core[i] = 0; }
    let big = LD ? open(and(LD.hole, blk), w, h, r) : new Uint8Array(N);
    if (any(core)) {
      const thick = open(blk, w, h, Math.max(r, Math.round(R * 0.6)));
      const C = label(thick, w, h), keep = new Uint8Array(C.n + 1);
      for (let i = 0; i < N; i++) if (core[i] && C.lab[i]) keep[C.lab[i]] = 1;
      for (let i = 0; i < N; i++) if (C.lab[i] && keep[C.lab[i]]) big[i] = 1;
      big = open(big, w, h, r);
    }
    // okraj postavy: souvislý pás dřeva kolem celé siluety; černá skrz jen uvnitř něj
    const rimPx = Math.max(r, Math.round(px(opt.rim == null ? 3 : opt.rim)));
    const rim = opt.rim === 0 || EX ? new Uint8Array(N) : andNot(sil, erode(sil, w, h, rimPx));
    big = open(andNot(big, rim), w, h, r);
    // kresba: barvy v roli Černá jsou díra hned (dostanou dělicí linky a okraj jako černé plochy z kresby)
    if (EX) { const bd = isRole("board"); if (any(bd)) big = or(big, open(and(smooth(bd, w, h, sig), sil), w, h, r)); }
    // linky mezi barvami: každá plocha jiné barvy (i černá díra) dostane dřevěné ohraničení
    let sep = new Uint8Array(N);
    if ((opt.sepLine == null ? 1.5 : opt.sepLine) > 0) {
      const Lc = new Int16Array(N);
      for (let i = 0; i < N; i++) Lc[i] = L[i] < 0 ? (EX ? 101 : -2) : roles[L[i]] === "outline" ? (big[i] ? 100 : -1) : L[i];
      // u kresby je černá díra samostatná "barva": vedle natřené plochy bez čáry dostane dřevěnou linku
      if (EX) for (let i = 0; i < N; i++) if (big[i]) Lc[i] = 100; else if (LD.line[i]) Lc[i] = -1;
      const b = new Uint8Array(N);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, a = Lc[i];
        if (a < 0) continue;
        if ((x + 1 < w && Lc[i + 1] >= 0 && Lc[i + 1] !== a) || (y + 1 < h && Lc[i + w] >= 0 && Lc[i + w] !== a)) b[i] = 1;
      }
      const half = Math.max(r, Math.round(px(opt.sepLine == null ? 1.5 : opt.sepLine) / 2));
      const blackLine = new Uint8Array(N);
      for (let i = 0; i < N; i++) blackLine[i] = Lc[i] === -1 ? 1 : 0;
      const dB = any(blackLine) ? edt(blackLine, w, h, false).d : null, limB = (half + 2) * (half + 2);
      if (dB) for (let i = 0; i < N; i++) if (b[i] && dB[i] <= limB) b[i] = 0;   // tady barvy už odděluje černá čára
      if (EX) for (let i = 0; i < N; i++) if (Lc[i] === 101) b[i] = 0;   // na straně pozadí nic
      if (EX) { // obrys proti pozadí plnou šířkou dovnitř
        const bo = new Uint8Array(N);   // všechny strany, nejen pravá a dolní
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const i = y * w + x, a = Lc[i];
          if (a < 0 || a === 101 || (a !== 100 && dB && dB[i] <= limB)) continue;
          if ((x + 1 < w && Lc[i + 1] === 101) || (x > 0 && Lc[i - 1] === 101) || (y + 1 < h && Lc[i + w] === 101) || (y > 0 && Lc[i - w] === 101)) bo[i] = 1;
        }
        const boH = new Uint8Array(N);   // černá plocha na kraji postavy: širší dřevěný okraj, ať silueta nesplyne s deskou
        for (let i = 0; i < N; i++) if (bo[i] && Lc[i] === 100) { boH[i] = 1; bo[i] = 0; }
        sep = or(sep, and(smooth(dilate(bo, w, h, 2 * half), w, h, Math.max(1, px(0.35))), sil));
        if (any(boH)) sep = or(sep, and(smooth(dilate(boH, w, h, Math.max(2 * half, rimPx)), w, h, Math.max(1, px(0.35))), sil));
      }
      sep = or(sep, and(smooth(dilate(b, w, h, half), w, h, Math.max(1, px(0.35))), sil));   // dělicí linka hladce, ne po pixelech
      big = open(andNot(big, sep), w, h, r);
    }
    { // barvy v roli Černá (deska): díra až na desku, kromě dělicích linek a okraje
      const bd = isRole("board");
      if (any(bd) && !EX) big = or(big, open(andNot(andNot(and(smooth(bd, w, h, sig), sil), sep), rim), w, h, r));
    }
    if (!EX) { const C = label(big, w, h), lim = (opt.minHole == null ? 2 + (opt.smooth == null ? 3 : opt.smooth) : opt.minHole) * 2 / (mm * mm);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < lim) big[i] = 0; }

    const l2 = andNot(sil, big);
    { // odtržené zbytky siluety (konce vousů apod.) zahodit
      const C = label(l2, w, h);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] * mm * mm < opt.minPiece * 5) { l2[i] = 0; sil[i] = 0; }
    }
    // horní vrstva: obrysy a dřevo
    let top;
    if (LD) {
      // horní vrstva = přesně kresba z nástroje Linky (+ dřevěné plochy), nic se nepřepočítává
      // horní vrstva = přesná kopie tahů z kresby, nic se nedoplňuje
      top = EX ? andNot(LD.line, big) : andNot(and(LD.line, sil), big);
      for (let i = 0; i < N; i++) if (L[i] >= 0 && roles[L[i]] === "wood" && sil[i]) top[i] = 1;
    } else {
    let colored = andNot(and(smooth(not(levAtLeast(NL)), w, h, sig), sil), big);
    colored = open(colored, w, h, r);
    top = andNot(andNot(sil, colored), big);
    if (opt.dropThin) top = open(top, w, h, r);
    else {
      // tenké tahy jen doplnit na minimální šířku, ne přifouknout
      const thin = andNot(top, open(top, w, h, r));
      top = andNot(or(top, and(dilate(thin, w, h, Math.max(1, Math.round(r * 0.5))), l2)), big);
    }
    }
    top = or(top, and(or(rim, sep), l2));
    if (EX) {
      // nic – tvar linek zůstává přesně podle kresby
    } else if (LD) {
      const holesIn = andNot(l2, top), C = label(holesIn, w, h), lim = (opt.minHole == null ? 3 : opt.minHole) / (mm * mm);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < lim) top[i] = 1;
    } else if ((opt.smooth == null ? 3 : opt.smooth) > 0) {
      const lv = opt.smooth == null ? 3 : opt.smooth;
      // jemné uhlazení bez rozbití sítě: Gauss + zaplnit drobné zářezy, tenké linky zůstanou
      // symetrické uhlazení (nepřifukuje), tenké linky zůstanou beze změny
      const thinT = andNot(top, open(top, w, h, r));
      let t2 = and(or(smooth(top, w, h, Math.max(1, px(0.1 + 0.07 * lv))), thinT), l2);
      // drobné otvory uvnitř sítě ("bubliny") zaplnit dřevem
      const holesIn = andNot(l2, t2), C = label(holesIn, w, h), lim = (opt.minHole == null ? 2 + lv : opt.minHole) / (mm * mm);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < lim) t2[i] = 1;
      top = or(t2, and(rim, l2));
    }

    // spojit oddělené ostrůvky horní vrstvy (oko v černé hlavě apod.) s hlavní sítí linek
    // nejkratší cestou uvnitř siluety; tím drží pohromadě každá vrstva a nic se nelepí
    // dělicí kroužky kolem ploch uvnitř jedné barvy (kamínky v hromádce): když jsou obě strany
    // ve stejné natírané vrstvě, kroužek není potřeba (hranici ukáže barevná linka) a nemusí se mostit
    if (EX) {
      // volně ležící tah (prst v rukavici, zahnutí pusy) nedrží s ostatními linkami: místo mostu štěrbina skrz na desku,
      // jako černá čára v předloze. Jen když nic neuzavírá (jinak by kus vrstvy vypadl).
      const C = label(or(top, and(ring, rect)), w, h);
      let main = 0;
      for (let i = 0; i < N; i++) if (ring[i] && C.lab[i]) { main = C.lab[i]; break; }
      const lim = 60 / (mm * mm), conv = new Uint8Array(C.n + 1);
      for (let q = 1; q <= C.n; q++) if (q !== main && C.sizes[q] <= lim) conv[q] = 1;
      if (conv.some((v) => v)) {
        const cm = new Uint8Array(N);
        for (let i = 0; i < N; i++) if (conv[C.lab[i]] && C.lab[i]) cm[i] = 1;
        // uzavřené tahy vynechat: co je obklopené tahem a není díra, by vypadlo
        const outside = new Uint8Array(N), st = [];
        for (let x = 0; x < w; x++) st.push(x, (h - 1) * w + x);
        for (let y = 0; y < h; y++) st.push(y * w, y * w + w - 1);
        while (st.length) { const i = st.pop(); if (outside[i] || cm[i]) continue; outside[i] = 1; const x = i % w; if (x > 0) st.push(i - 1); if (x < w - 1) st.push(i + 1); if (i >= w) st.push(i - w); if (i + w < N) st.push(i + w); }
        const encl = new Float64Array(C.n + 1), own = edt(cm, w, h, true).idx;
        for (let i = 0; i < N; i++) if (!outside[i] && !cm[i] && !big[i]) encl[C.lab[own[i]]]++;
        for (let i = 0; i < N; i++) { const q = C.lab[i]; if (q && conv[q] && encl[q] * mm * mm <= 2) { top[i] = 0; if (sil[i]) { big[i] = 1; l2[i] = 0; } } }
      }
    }
    if (EX) {
      const C = label(top, w, h);
      let main = 0;
      for (let i = 0; i < N; i++) if (ring[i] && C.lab[i]) { main = C.lab[i]; break; }
      if (!main) for (let q = 1; q <= C.n; q++) if (!main || C.sizes[q] > C.sizes[main]) main = q;
      const isLine = LD.line;
      const lvSet = new Map();
      const near = dilate(top, w, h, 2);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!near[i] || top[i] || L[i] < 0) continue;
        // najít komponentu sousední linky
        for (const j of [i - 2, i + 2, i - 2 * w, i + 2 * w, i - 1, i + 1, i - w, i + w]) {
          if (j < 0 || j >= N || !C.lab[j] || C.lab[j] === main) continue;
          const q = C.lab[j];
          if (!lvSet.has(q)) lvSet.set(q, new Set());
          lvSet.get(q).add(lev[i]);
        }
      }
      const drop = new Uint8Array(C.n + 1), fromLine = new Uint8Array(C.n + 1);
      for (let i = 0; i < N; i++) if (C.lab[i] && isLine[i]) fromLine[C.lab[i]] = 1;
      for (const [q, set] of lvSet) if (set.size === 1 && !fromLine[q]) drop[q] = 1;
      for (let i = 0; i < N; i++) if (C.lab[i] && drop[C.lab[i]]) top[i] = 0;
    }
    let nBridges = 0;
    if (opt.bridges !== false) {
      const half = Math.max(r, Math.round(px(opt.sepLine == null ? 1.5 : opt.sepLine || 1.5) / 2));
      for (let round = 0; round < 12; round++) {
        const C = label(top, w, h);
        if (C.n <= 1) break;
        let main = 1;
        for (let q = 2; q <= C.n; q++) if (C.sizes[q] > C.sizes[main]) main = q;
        // BFS od hlavní sítě přes siluetu
        const dist = new Int32Array(N).fill(-1), par = new Int32Array(N).fill(-1), queue = new Int32Array(N);
        let qh = 0, qt = 0;
        for (let i = 0; i < N; i++) if (C.lab[i] === main) { dist[i] = 0; queue[qt++] = i; }
        while (qh < qt) {
          const i = queue[qh++], x = i % w, y = (i / w) | 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            const j = yy * w + xx;
            if (dist[j] >= 0 || !sil[j]) continue;
            dist[j] = dist[i] + 1; par[j] = i;
            if (!C.lab[j] || C.lab[j] === main) queue[qt++] = j;   // přes jiný ostrůvek se nepokračuje, ten se připojí sám
          }
        }
        const best = new Int32Array(C.n + 1).fill(-1);
        for (let i = 0; i < N; i++) { const q = C.lab[i]; if (q && q !== main && dist[i] > 0 && (best[q] < 0 || dist[i] < dist[best[q]])) best[q] = i; }
        const path = new Uint8Array(N);
        let added = 0;
        for (let q = 1; q <= C.n; q++) {
          if (q === main || best[q] < 0) continue;
          for (let i = best[q]; i >= 0 && C.lab[i] !== main; i = par[i]) path[i] = 1;
          added++;
        }
        if (!added) break;
        const br = and(dilate(path, w, h, half), sil);
        top = or(top, br);
        big = andNot(big, br);
        for (let i = 0; i < N; i++) if (br[i]) l2[i] = 1;
        nBridges += added;
      }
    }

    // spojky postavy s rámečkem: na každé straně z nejbližšího silného místa postavy rovný pásek k rámečku
    const nStruts = Math.max(0, Math.min(4, opt.struts == null ? 4 : opt.struts));
    if (nStruts > 0 && f > 0) {
      const sw = Math.max(r * 2, Math.round(px(opt.strutWidth || 4)));
      let core = erode(sil, w, h, Math.ceil(sw / 2));
      if (!any(core)) core = sil;
      const strut = new Uint8Array(N);
      let best = { L: null, R: null, B: null, T: null };
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        if (!core[y * w + x]) continue;
        if (!best.L || x < best.L[0]) best.L = [x, y];
        if (!best.R || x > best.R[0]) best.R = [x, y];
        if (!best.T || y < best.T[1]) best.T = [x, y];
        if (!best.B || y > best.B[1]) best.B = [x, y];
      }
      const fill = (x0, x1, y0, y1) => { for (let y = Math.max(0, y0); y <= Math.min(h - 1, y1); y++) for (let x = Math.max(0, x0); x <= Math.min(w - 1, x1); x++) strut[y * w + x] = 1; };
      const hw = Math.floor(sw / 2);
      ["L", "R", "B", "T"].slice(0, nStruts).forEach((side) => {
        const p = best[side]; if (!p) return;
        if (side === "L") fill(f - 2, p[0], p[1] - hw, p[1] + hw);
        if (side === "R") fill(p[0], w - f + 1, p[1] - hw, p[1] + hw);
        if (side === "T") fill(p[0] - hw, p[0] + hw, f - 2, p[1]);
        if (side === "B") fill(p[0] - hw, p[0] + hw, p[1], h - f + 1);
      });
      const st = and(strut, rect);
      top = or(top, st);
      big = andNot(big, st);
      for (let i = 0; i < N; i++) if (st[i]) l2[i] = 1;
    }

    // gravírování černých detailů: čáry mezi dvěma dřevěnými plochami (rukavice/obličej),
    // zorničky a jiné silné černé skvrny, které nejdou skrz. Čáry vedle barvy zůstávají čistým dřevem.
    let engrave = new Uint8Array(N);
    if (opt.engrave !== false) {
      const O = andNot(and(isRole("outline"), sil), big);
      const paintM = new Uint8Array(N), woodM = new Uint8Array(N);
      for (let i = 0; i < N; i++) if (L[i] >= 0 && sil[i]) { if (typeof roles[L[i]] === "number") paintM[i] = 1; else if (roles[L[i]] === "wood") woodM[i] = 1; }
      const dP = any(paintM) ? edt(paintM, w, h, false).d : null, dW = any(woodM) ? edt(woodM, w, h, false).d : null;
      const T = px((opt.sepLine == null ? 1.5 : opt.sepLine) / 2 + opt.minFeature), T2 = T * T;
      const blob = open(O, w, h, Math.max(r + 1, Math.round(px(1.5))));
      for (let i = 0; i < N; i++) {
        if (!O[i] || !top[i] || rim[i]) continue;
        const onWood = dW && (!dP || dW[i] <= dP[i]) && (!dP || dP[i] > T2);
        if (onWood || blob[i]) engrave[i] = 1;
      }
      engrave = and(close(engrave, w, h, Math.max(2, Math.round(px(1.5)))), and(O, top));   // spojit přerušované úseky jen po černé čáře
      engrave = andNot(engrave, rim);
      engrave = open(engrave, w, h, 1);
      const C = label(engrave, w, h);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] * mm * mm < 0.5) engrave[i] = 0;
    }
    const hasEngrave = any(engrave);

    // náhled "linky": síť linek tmavě, plochy světle v jejich barvě, černé díry šedě
    const linesImg = new Uint8ClampedArray(N * 4);
    for (let i = 0; i < N; i++) {
      let c;
      if (!rect[i]) c = [0, 0, 0, 0];
      else if (engrave[i]) c = [20, 14, 10, 255];
      else if (top[i] || ring[i]) c = [70, 48, 30, 255];
      else if (big[i]) c = [150, 150, 152, 255];
      else if (sil[i] && L[i] >= 0) { const q = cent[L[i]]; c = [255 - (255 - q[0]) * 0.55, 255 - (255 - q[1]) * 0.55, 255 - (255 - q[2]) * 0.55, 255]; }
      else c = [255, 255, 255, 255];
      linesImg.set(c, i * 4);
    }

    // masky vrstev 1..NL (index 0 = deska)
    const M = [rect, l2];
    for (let k = 3; k < NL; k++) {
      let m = andNot(and(smooth(levAtLeast(k), w, h, sig), sil), big);
      m = open(m, w, h, r);
      M.push(and(or(m, top), M[M.length - 1]));
    }
    M.push(and(top, M[M.length - 1]));

    // kousky: oddělené díly vrstev 3..NL, značka na vrstvě pod nimi
    const pieces = M.map(() => null);
    let nPieces = 0;
    for (let k = 2; k < NL; k++) {
      const m = M[k];
      let C = label(m, w, h);
      for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] * mm * mm < opt.minPiece) m[i] = 0;
      C = label(m, w, h);
      let main = 0;
      for (let q = 1; q <= C.n; q++) if (!main || C.sizes[q] > C.sizes[main]) main = q;
      const pc = new Uint8Array(N);
      for (let i = 0; i < N; i++) pc[i] = C.lab[i] && C.lab[i] !== main ? 1 : 0;
      pieces[k] = C.n > 1 ? pc : null;
      nPieces += Math.max(0, C.n - 1);
    }
    for (let k = 3; k < NL; k++) M[k] = and(M[k], or(M[k - 1], new Uint8Array(N)));

    const wmm = w * mm, hmm = h * mm;
    const cont = (m) => (curveSmooth > 0 ? contoursSub(m, w, h, 0.9) : contours(m, w, h, eps));
    const ringR = and(ring, rect), cRing = cont(ringR);
    const cM = M.map((m, k) => (k === 0 ? cont(rect) : cont(m)));
    // plochy k natření na každé vrstvě a jejich ohraničení v barvě nátěru
    const nearAny = (() => { const m = new Uint8Array(N); for (let i = 0; i < N; i++) if (L[i] >= 0 && typeof roles[L[i]] === "number") m[i] = 1; return any(m) ? edt(m, w, h, true).idx : null; })();
    const nearFor = (k) => {
      const pm = new Uint8Array(N);
      for (let i = 0; i < N; i++) pm[i] = L[i] >= 0 && typeof roles[L[i]] === "number" && lvlOf[L[i]] === k + 1 ? 1 : 0;
      return any(pm) ? edt(pm, w, h, true).idx : nearAny;
    };
    const ewP = Math.max(1, Math.round(Math.max(w, h) / 650));
    const paintInfo = M.map((m0, k) => {
      if (k === 0 || k >= NL - 1 || !nearAny) return null;
      const near = nearFor(k), above = M[k + 1];
      const mat = or(ringR, m0);
      const region = andNot(andNot(m0, above), ringR);
      const dOut = edt(not(mat), w, h, false).d, lim = (1.5 + ewP) * (1.5 + ewP);
      const colorOf = new Int16Array(N).fill(-1);
      for (let i = 0; i < N; i++) if (region[i]) colorOf[i] = L[near[i]];
      const groups = [], lineMask = new Uint8Array(N);
      for (let q = 0; q < cent.length; q++) {
        const mq = new Uint8Array(N); let cnt = 0;
        for (let i = 0; i < N; i++) if (colorOf[i] === q) { mq[i] = 1; cnt++; }
        if (cnt * mm * mm < 4) continue;
        const lines = [];
        for (const poly of contours(mq, w, h, -1)) {
          let cur = [];
          const flush = () => { if (cur.length >= 2) { const l = rdpOpen(cur, eps); if (Math.hypot(l[0][0] - l[l.length - 1][0], l[0][1] - l[l.length - 1][1]) > 2 || l.length > 2) lines.push(l); } cur = []; };
          for (const pt of poly.concat([poly[0]])) {
            const x = Math.min(w - 1, Math.max(0, Math.round(pt[0]))), y = Math.min(h - 1, Math.max(0, Math.round(pt[1])));
            if (dOut[y * w + x] <= lim) flush(); else cur.push(pt);
          }
          flush();
        }
        if (lines.length) groups.push({ q, color: "#" + cent[q].map((v) => v.toString(16).padStart(2, "0")).join(""), lines });
        const er = erode(mq, w, h, ewP);
        for (let i = 0; i < N; i++) if (mq[i] && !er[i] && dOut[i] > lim) lineMask[i] = 1;
      }
      return { colorOf, groups, lineMask };
    });

    const sheets = M.map((m, k) => {
      const mark = k === 0 ? cM[1] : pieces[k + 1] ? cont(pieces[k + 1]) : [];
      const label = k === 0 ? "1 Deska - natřít černě" : k === NL - 1 ? `${k + 1} Čisté dřevo` : `${k + 1} Barvy - natřít podle náhledu`;
      return { label: k === NL - 1 && hasEngrave ? label + " + gravírovat fialové" : label, cut: k === 0 ? cM[0] : cont(or(ringR, m)), mark, paint: paintInfo[k] ? paintInfo[k].groups : [], engrave: k === NL - 1 && hasEngrave ? cont(and(engrave, m)) : [] };
    });
    const files = sheets.map((sh) => ({ svg: svg(wmm, hmm, mm, sh.cut, sh.mark, "vrstva " + sh.label, sh.paint, sh.engrave) }));
    const all = combined(wmm, hmm, mm, sheets);

    // obrázky vrstev: dřevo s propálenou hranou, nátěr, díry průhledné, azurové značky
    const ew = Math.max(1, Math.round(Math.max(w, h) / 650));
    const edgeOf = (m) => andNot(m, erode(m, w, h, ew));
    const CYAN = [0, 190, 205, 255], BURN = [120, 78, 44, 255], BLACK = [30, 30, 34, 255];
    const layers = M.map((m0, k) => {
      const data = new Uint8ClampedArray(N * 4);
      const mat = k === 0 ? rect : or(ringR, m0);
      const edge = edgeOf(mat);
      const markM = k === 0 ? edgeOf(M[1]) : pieces[k + 1] ? edgeOf(pieces[k + 1]) : null;
      const above = k + 1 < M.length ? M[k + 1] : null;
      const pinf = paintInfo[k];
      const wood = k === NL - 1 ? [232, 204, 160, 255] : [216 + 4 * k, 184 + 4 * k, 138 + 4 * k, 255];
      for (let i = 0; i < N; i++) {
        if (!mat[i]) continue;
        let c;
        if (k === 0) c = edge[i] ? [10, 10, 12, 255] : markM[i] ? CYAN : BLACK;
        else if (edge[i]) c = BURN;
        else if (markM && markM[i]) c = CYAN;
        else if (pinf && pinf.colorOf[i] >= 0) { const cc = cent[pinf.colorOf[i]]; c = pinf.lineMask[i] ? [cc[0] * 0.6, cc[1] * 0.6, cc[2] * 0.6, 255] : cc.concat(255); }
        else if (k === NL - 1 && engrave[i]) c = [58, 36, 22, 255];
        else c = wood;
        data.set(c, i * 4);
      }
      return { w, h, data };
    });
    const holes = label(big, w, h).n;
    return {
      files, all, layers, mm, lines: { w, h, data: linesImg },
      cutPx: sheets.map((sh) => sh.cut.map((p) => closedD(p, 1, 0, 0)).join(" ")),
      stats: { wmm, hmm, silParts: label(l2, w, h).n, holes, pieces: nPieces, layers: NL, bridges: nBridges, frameJoined: f > 0 ? label(or(ringR, M[NL - 1]), w, h).n === 1 : true },
    };
  }

  // ---------- samostatná funkce: obrázek -> barevné křivky (SVG) ----------
  function smoothPath(poly, eps, cornerRad, passes, scale, ox, oy) {
    let p = poly, n = p.length;
    if (n < 4) return "";
    passes = passes == null ? 2 : passes; scale = scale || 1; ox = ox || 0; oy = oy || 0;
    for (let pass = 0; pass < passes; pass++) {
      const q = new Array(n);
      for (let i = 0; i < n; i++) { const a = p[(i - 1 + n) % n], b = p[i], c = p[(i + 1) % n]; q[i] = [(a[0] + 2 * b[0] + c[0]) / 4, (a[1] + 2 * b[1] + c[1]) / 4]; }
      p = q;
    }
    const q = rdp(p.concat([p[0]]), eps), m = q.length;
    if (m < 3) return "";
    // dlouhé rovné úseky zůstanou rovné (rámeček, spojky), jejich konce jsou rohy
    const longSeg = q.map((b, i) => { const c = q[(i + 1) % m]; return Math.hypot(c[0] - b[0], c[1] - b[1]) > 40; });
    const corner = q.map((b, i) => {
      if (longSeg[i] || longSeg[(i - 1 + m) % m]) return true;
      const a = q[(i - 1 + m) % m], c = q[(i + 1) % m];
      const v1x = b[0] - a[0], v1y = b[1] - a[1], v2x = c[0] - b[0], v2y = c[1] - b[1];
      const l = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y) || 1;
      return Math.acos(Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / l))) > cornerRad;
    });
    const fx = (v) => (v * scale + ox).toFixed(3), fy = (v) => (v * scale + oy).toFixed(3);
    let d = "M" + fx(q[0][0]) + " " + fy(q[0][1]);
    for (let i = 0; i < m; i++) {
      const pa = q[(i - 1 + m) % m], a = q[i], b = q[(i + 1) % m], nb = q[(i + 2) % m];
      if (longSeg[i]) { d += "L" + fx(b[0]) + " " + fy(b[1]); continue; }
      const c1 = corner[i] ? [a[0] + (b[0] - a[0]) / 3, a[1] + (b[1] - a[1]) / 3] : [a[0] + (b[0] - pa[0]) / 6, a[1] + (b[1] - pa[1]) / 6];
      const c2 = corner[(i + 1) % m] ? [b[0] - (b[0] - a[0]) / 3, b[1] - (b[1] - a[1]) / 3] : [b[0] - (nb[0] - a[0]) / 6, b[1] - (nb[1] - a[1]) / 6];
      d += "C" + fx(c1[0]) + " " + fy(c1[1]) + " " + fx(c2[0]) + " " + fy(c2[1]) + " " + fx(b[0]) + " " + fy(b[1]);
    }
    return d + "Z";
  }

  function vectorize(rgba, W, H, o) {
    o = o || {};
    const N = W * H, k = Math.max(2, Math.min(32, o.colors || 8));
    const fg = o.removeBg === false ? new Uint8Array(N).fill(1) : foreground(rgba, W, H, o.bg);
    // vyhlazení šumu (box blur RGB) jen pro určení barev
    let px = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) { px[i * 3] = rgba[i * 4]; px[i * 3 + 1] = rgba[i * 4 + 1]; px[i * 3 + 2] = rgba[i * 4 + 2]; }
    const br = Math.round(o.blur || 0);
    if (br > 0) {
      const tmp = new Float32Array(N * 3);
      for (let pass = 0; pass < 2; pass++) {
        const src = pass ? tmp : px, dst = pass ? px : tmp;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          let r = 0, g = 0, b = 0, c = 0;
          for (let j = -br; j <= br; j++) {
            const xx = pass ? x : Math.min(W - 1, Math.max(0, x + j)), yy = pass ? Math.min(H - 1, Math.max(0, y + j)) : y;
            const ii = yy * W + xx; r += src[ii * 3]; g += src[ii * 3 + 1]; b += src[ii * 3 + 2]; c++;
          }
          const ii = y * W + x; dst[ii * 3] = r / c; dst[ii * 3 + 1] = g / c; dst[ii * 3 + 2] = b / c;
        }
      }
    }
    let cnt = 0;
    for (let i = 0; i < N; i++) cnt += fg[i];
    if (!cnt) return { svg: "", colors: [], fg };
    const cent = dedupe(kmeans(flatSample((i, ch) => px[i * 3 + ch], fg, W, H, 50000), k, 12), o.mergeThr == null ? 18 : o.mergeThr), kk = cent.length;
    const lab = new Int16Array(N).fill(-1);
    for (let i = 0; i < N; i++) {
      if (!fg[i]) continue;
      let best = 0, bd = Infinity;
      for (let c = 0; c < kk; c++) { const dr = px[i * 3] - cent[c][0], dg = px[i * 3 + 1] - cent[c][1], db = px[i * 3 + 2] - cent[c][2]; const d = dr * dr + dg * dg + db * db; if (d < bd) { bd = d; best = c; } }
      lab[i] = best;
    }
    // skvrny menší než minArea přebarvit na nejbližší větší plochu
    const minPx = Math.max(1, o.minArea == null ? 12 : o.minArea);
    const bad = new Uint8Array(N);
    for (let c = 0; c < kk; c++) {
      const m = new Uint8Array(N);
      for (let i = 0; i < N; i++) m[i] = lab[i] === c ? 1 : 0;
      const L = label(m, W, H);
      for (let i = 0; i < N; i++) if (L.lab[i] && L.sizes[L.lab[i]] < minPx) bad[i] = 1;
    }
    if (any(bad)) {
      const good = new Uint8Array(N);
      for (let i = 0; i < N; i++) good[i] = lab[i] >= 0 && !bad[i] ? 1 : 0;
      if (any(good)) { const { idx } = edt(good, W, H, true); for (let i = 0; i < N; i++) if (bad[i]) lab[i] = lab[idx[i]]; }
    }
    // křivky po barvách, větší plochy dospod
    const area = new Array(kk).fill(0);
    for (let i = 0; i < N; i++) if (lab[i] >= 0) area[lab[i]]++;
    const order = area.map((a, c) => c).filter((c) => area[c] > 0).sort((a, b) => area[b] - area[a]);
    const detail = Math.max(1, Math.min(10, o.detail || 6));
    const eps = 3.2 - detail * 0.28, cornerRad = (55 * Math.PI) / 180;
    let x0 = W, x1 = -1, y0 = H, y1 = -1;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (lab[y * W + x] >= 0) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    const mmpx = (o.widthMm || 300) / Math.max(1, x1 - x0 + 1);
    const hex = (c) => "#" + c.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
    let body = "", total = 0;
    for (const c of order) total += area[c];
    const colors = [];
    for (const c of order) {
      const m = new Uint8Array(N);
      for (let i = 0; i < N; i++) m[i] = lab[i] === c ? 1 : 0;
      const d = contours(m, W, H, -1).map((pl) => smoothPath(pl, eps, cornerRad)).filter(Boolean).join("");
      if (!d) continue;
      const h = hex(cent[c]);
      body += `<path fill="${h}" stroke="${h}" stroke-width="0.8" stroke-linejoin="round" fill-rule="evenodd" d="${d}"/>\n`;
      colors.push({ hex: h, share: area[c] / total });
    }
    const svgOut = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${(W * mmpx).toFixed(2)}mm" height="${(H * mmpx).toFixed(2)}mm" viewBox="0 0 ${W} ${H}"><title>obrázek v křivkách</title>\n${body}</svg>\n`;
    return { svg: svgOut, colors, fg, W, H };
  }

  // ---------- nástroj Linky: kresba, černé plochy, políčka mezi linkami ----------
  function label4(m, W, H) {
    const lab = new Int32Array(m.length), sizes = [0], stack = new Int32Array(m.length);
    let n = 0;
    for (let s0 = 0; s0 < m.length; s0++) {
      if (!m[s0] || lab[s0]) continue;
      n++; let sp = 0, cnt = 0; stack[sp++] = s0; lab[s0] = n;
      while (sp) {
        const i = stack[--sp]; cnt++;
        const x = i % W;
        if (x > 0 && m[i - 1] && !lab[i - 1]) { lab[i - 1] = n; stack[sp++] = i - 1; }
        if (x < W - 1 && m[i + 1] && !lab[i + 1]) { lab[i + 1] = n; stack[sp++] = i + 1; }
        if (i >= W && m[i - W] && !lab[i - W]) { lab[i - W] = n; stack[sp++] = i - W; }
        if (i + W < m.length && m[i + W] && !lab[i + W]) { lab[i + W] = n; stack[sp++] = i + W; }
      }
      sizes.push(cnt);
    }
    return { lab, n, sizes };
  }

  // o: { widthMm, hole (mm), minCell (mm2) }
  function lineArt(an, roles, o) {
    const { W, H, lab, cent } = an, N = W * H;
    let x0 = W, x1 = -1;
    for (let i = 0; i < N; i++) if (lab[i] >= 0) { const x = i % W; if (x < x0) x0 = x; if (x > x1) x1 = x; }
    const mm = (o.widthMm || 300) / Math.max(1, x1 - x0 + 1), px = (v) => v / mm;
    const isOut = (k) => k >= 0 && roles[k] === "outline";
    let outlineIdx = roles.findIndex((r) => r === "outline");
    const hasOutline = outlineIdx >= 0;   // bez obrysové barvy (Krtek) se žádná barva v políčku nepřepisuje
    if (outlineIdx < 0) outlineIdx = 0;
    // tahy
    let line = new Uint8Array(N);
    for (let i = 0; i < N; i++) line[i] = isOut(lab[i]) ? 1 : 0;
    line = smooth(line, W, H, 0.8);
    // tahy smazané štětcem
    if (o.erase) for (let i = 0; i < N; i++) if (o.erase[i]) line[i] = 0;
    // políčka mezi linkami
    const fgFree = new Uint8Array(N);
    for (let i = 0; i < N; i++) fgFree[i] = lab[i] >= 0 && !line[i] ? 1 : 0;
    let C = label4(fgFree, W, H);
    // drobná políčka (mezi chlupy, ve spojích) přidat k linkám
    const minCell = (o.minCell == null ? 3 : o.minCell) / (mm * mm);
    let changed = false;
    for (let i = 0; i < N; i++) if (C.lab[i] && C.sizes[C.lab[i]] < minCell) { line[i] = 1; fgFree[i] = 0; changed = true; }
    if (changed) C = label4(fgFree, W, H);
    // barva políčka = převažující barva; políčko z více velkých barev (oko bez čáry) zůstane po pixelech
    const nC = C.n, k = cent.length, hist = new Int32Array((nC + 1) * k);
    for (let i = 0; i < N; i++) if (C.lab[i] && lab[i] >= 0 && !isOut(lab[i])) hist[C.lab[i] * k + lab[i]]++;
    const dom = new Int16Array(nC + 1).fill(-1), multi = new Uint8Array(nC + 1);
    for (let c = 1; c <= nC; c++) {
      let b1 = -1, v1 = 0, v2 = 0;
      for (let j = 0; j < k; j++) { const v = hist[c * k + j]; if (v > v1) { v2 = v1; v1 = v; b1 = j; } else if (v > v2) v2 = v; }
      dom[c] = b1;
      if (v2 > Math.max(0.12 * C.sizes[c], 4 / (mm * mm))) multi[c] = 1;
    }
    // černé plochy (díry na desku): místa výrazně silnější než běžný tah kresby (zornice, čumák, uši Mickeyho, tělo Krtka).
    // Běžné tahy zůstanou dřevěnou linkou, i když jsou silné (Mickey). Tvar díry = sjednocení kruhů, které se do černé vejdou,
    // takže díra nepoleze po navazujících tazích.
    const r = Math.max(1, Math.round(px(1.2) / 2));
    let hole = new Uint8Array(N), strokeMm = 0;
    if (o.autoHole && any(line)) {
      const d = edt(not(line), W, H, false).d, vals = [];
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const i = y * W + x, v = d[i];
        if (!line[i] || v < 1) continue;
        if (v >= d[i - 1] && v >= d[i + 1] && v >= d[i - W] && v >= d[i + W]) vals.push(Math.sqrt(v));
      }
      vals.sort((p, q) => p - q);
      const sw = vals.length ? 2 * vals[Math.floor(vals.length / 2)] : 0;   // typická šířka tahu v px
      strokeMm = sw * mm;
      const R = Math.max(r + 1, Math.round(px(o.hole || 3) / 2), Math.round(0.8 * sw));
      const core = open(line, W, H, R);
      { const L2 = label(core, W, H), minA = 2 * Math.PI * R * R; for (let i = 0; i < N; i++) if (L2.lab[i] && L2.sizes[L2.lab[i]] < minA) core[i] = 0; }
      if (any(core)) hole = and(dilate(core, W, H, Math.max(1, Math.round(R * 0.35))), line);
      // ztluštění v uzlu, kde se sbíhá víc tahů, není černá plocha: jeho okraj z velké části navazuje na další tahy.
      // Zornice, čumák nebo ucho mají okolo hlavně barvu nebo pozadí.
      if (any(hole)) {   // nejsilnější místo plochy musí být aspoň 2,4× silnější než běžný tah (jinak je to jen silnější kus tahu)
        const L6 = label(hole, W, H), mx = new Float64Array(L6.n + 1), rin = Math.max(1.2 * sw, px(o.hole || 3) / 2);
        for (let i = 0; i < N; i++) if (L6.lab[i] && d[i] > mx[L6.lab[i]]) mx[L6.lab[i]] = d[i];
        // a nesmí to být protáhlý tah (silná linka podrážky): plocha nejvýš ~4 kruhy o největší síle
        for (let i = 0; i < N; i++) { const q = L6.lab[i]; if (q && (Math.sqrt(mx[q]) < rin || (Math.sqrt(mx[q]) < 2.5 * sw && L6.sizes[q] > 4 * Math.PI * mx[q]))) hole[i] = 0; }
      }
      if (any(hole)) {
        const L5 = label(hole, W, H), toLine = new Int32Array(L5.n + 1), tot = new Int32Array(L5.n + 1);
        const ring = andNot(dilate(hole, W, H, 1), hole), owner = edt(hole, W, H, true).idx;
        for (let i = 0; i < N; i++) {
          if (!ring[i]) continue;
          const q = L5.lab[owner[i]];
          tot[q]++; if (line[i]) toLine[q]++;
        }
        for (let i = 0; i < N; i++) { const q = L5.lab[i]; if (q && toLine[q] > 0.4 * tot[q]) hole[i] = 0; }
      }
    }
    return { W, H, mm, line, hole, cells: C.lab, nCells: nC, cellSize: C.sizes, dom, multi, outlineIdx, hasOutline, strokeMm };
  }

  // složit data pro vrstvy: úpravy uživatele (barva políčka, štětec černá plocha / obrys)
  function composeLines(an, la, cellColor, holeEdit) {
    const N = la.W * la.H, lab = new Int8Array(N).fill(-1), hole = new Uint8Array(N);
    for (let i = 0; i < N; i++) {
      if (an.lab[i] < 0) continue;
      if (la.line[i]) {
        lab[i] = la.outlineIdx;
        const e = holeEdit ? holeEdit[i] : 0;
        hole[i] = e > 0 ? 1 : e < 0 ? 0 : la.hole[i];
        continue;
      }
      const c = la.cells[i];
      if (!c) { lab[i] = an.lab[i]; continue; }
      const ov = cellColor ? cellColor.get(c) : undefined;
      lab[i] = ov !== undefined ? ov : la.multi[c] && an.lab[i] >= 0 && (!la.hasOutline || an.lab[i] !== la.outlineIdx) ? an.lab[i] : la.dom[c] >= 0 ? la.dom[c] : an.lab[i];
    }
    return { lab, line: la.line, hole };
  }

  // linky jako vektorová kresba (SVG): černé tahy a plochy hladkými křivkami
  function linesSvg(la, holeEdit, widthMm) {
    const N = la.W * la.H, m = new Uint8Array(N);
    for (let i = 0; i < N; i++) m[i] = la.line[i];
    const prev = curveSmooth; curveSmooth = 3;
    const d = contoursSub(m, la.W, la.H, 0.9).map((p) => closedD(p, la.mm, 0, 0)).join(" ");
    curveSmooth = prev;
    const wmm = la.W * la.mm, hmm = la.H * la.mm;
    return `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${wmm.toFixed(2)}mm" height="${hmm.toFixed(2)}mm" viewBox="0 0 ${wmm.toFixed(3)} ${hmm.toFixed(3)}"><title>linky</title>\n<path fill="#000" fill-rule="evenodd" d="${d}"/>\n</svg>\n`;
  }

  root.LaserLayers = { analyze, build, vectorize, lineArt, composeLines, linesSvg };
  if (typeof module !== "undefined") module.exports = root.LaserLayers;
})(typeof window !== "undefined" ? window : globalThis);
