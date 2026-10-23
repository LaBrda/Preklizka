"""Výpočet vrstev pro laser ve vyšším rozlišení (server).

Vstup je stejný, jaký si připraví webové ovládání: mapa barev (index barvy na pixel, -1 = pozadí),
paleta, role barev, maska linek a maska černých ploch. Výpočet běží ve zvětšeném rozlišení
s hladkými hranicemi barev (každá barva se rozmaže a vyhraje ta nejsilnější), takže řezy
nemají schody po pixelech. Výsledkem je SVG pro laser a náhledy vrstev.
"""
from __future__ import annotations

import base64
import io
import math

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CUT, MARK, ENGRAVE = "#000000", "#00e0e0", "#ff00ff"


# ---------------------------------------------------------------- pomocné operace s maskami
def disk(r: int) -> np.ndarray:
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def dilate(m, r):
    return m if r < 1 else cv2.dilate(m.astype(np.uint8), disk(r)).astype(bool)


def erode(m, r):
    return m if r < 1 else cv2.erode(m.astype(np.uint8), disk(r)).astype(bool)


def opening(m, r):
    return dilate(erode(m, r), r)


def closing(m, r):
    return erode(dilate(m, r), r)


def smooth(m, sigma):
    if sigma <= 0:
        return m
    f = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    return f > 0.5


def nearest(mask):
    """Vzdálenost (float32) a souřadnice nejbližšího pixelu masky (int32) – úsporně přes OpenCV."""
    src = (~mask).astype(np.uint8)
    dist, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(src == 0)          # pořadí nulových pixelů = čísla štítků (od 1)
    lut_y = np.concatenate([[0], ys]).astype(np.int32)
    lut_x = np.concatenate([[0], xs]).astype(np.int32)
    return dist, lut_y[lab], lut_x[lab]


def drop_small(m, min_px):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


# ---------------------------------------------------------------- obrysy -> SVG křivky
SMOOTH = {"passes": 6, "eps": 0.9}


def contours(m: np.ndarray, scale: float, ox: float = 0, oy: float = 0, smooth_passes: int | None = None, eps: float | None = None):
    """Uzavřené obrysy masky jako seznam polí bodů (už v mm, posunuté o ox, oy)."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        p = c[:, 0, :].astype(np.float64) + 0.5
        n = len(p)
        smooth_passes = SMOOTH["passes"] if smooth_passes is None else smooth_passes
        eps = SMOOTH["eps"] if eps is None else eps
        if n < 6:
            continue
        for _ in range(smooth_passes):  # vyhlazení bodů (odstraní schodky po pixelech)
            p = (np.roll(p, 1, 0) + 2 * p + np.roll(p, -1, 0)) / 4
        q = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2), eps, True)[:, 0, :].astype(np.float64)
        if len(q) < 3:
            continue
        out.append(q * scale + [ox, oy])
    return out


def path_d(polys, curves=True):
    """Polygony -> SVG cesta. Hladké křivky (Catmull-Rom -> Bézier), dlouhé rovné úseky zůstanou rovné."""
    parts = []
    for p in polys:
        n = len(p)
        if not curves or n < 4:
            parts.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in p) + "z")
            continue
        seg = np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)
        straight = seg > max(4.0, np.median(seg) * 6)
        d = [f"M{p[0][0]:.3f},{p[0][1]:.3f}"]
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            if straight[i]:
                d.append(f"L{b[0]:.3f},{b[1]:.3f}")
                continue
            pa, nb = p[(i - 1) % n], p[(i + 2) % n]
            c1 = a + (b - pa) / 6 if not straight[(i - 1) % n] else a + (b - a) / 3
            c2 = b - (nb - a) / 6 if not straight[(i + 1) % n] else b - (b - a) / 3
            d.append(f"C{c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {b[0]:.3f},{b[1]:.3f}")
        parts.append("".join(d) + "Z")
    return " ".join(parts)


def open_lines_d(lines):
    return " ".join("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in l) for l in lines if len(l) >= 2)


# ---------------------------------------------------------------- hlavní výpočet
def build(lab: np.ndarray, cent: list, roles: list, opt: dict, line: np.ndarray | None = None, hole: np.ndarray | None = None) -> dict:
    H0, W0 = lab.shape
    line = np.zeros_like(lab, bool) if line is None else line.astype(bool)
    hole = np.zeros_like(lab, bool) if hole is None else hole.astype(bool)
    fg0 = (lab >= 0) | line
    ys, xs = np.where(fg0)
    if len(xs) == 0:
        raise ValueError("Obrázek je prázdný (po odstranění pozadí nic nezbylo).")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    width = float(opt.get("width", 300))
    mm0 = width / (x1 - x0 + 1)                      # mm na pixel původního obrázku
    NL = int(max(3, min(5, opt.get("layers", 3))))
    frame, gap = float(opt.get("frame", 10)), float(opt.get("gap", -3))

    # zvětšení: pracovní rozlišení kolem 3000 px
    # pracovní rozlišení: nejvýš ~4 Mpx (paměť pod ~700 MB), malé obrázky se zvětší
    U = float(max(1.0, min(4.0, math.sqrt(4.0e6 / float(W0 * H0)))))
    lv = int(max(0, min(5, opt.get("smooth", 3))))
    SMOOTH["passes"] = int(2 + (1 + lv) * U)
    SMOOTH["eps"] = 0.3 + 0.12 * lv * U
    mm = mm0 / U
    px = lambda v: v / mm                            # mm -> pracovní pixely
    pad = max(0, int(round(px(frame + gap))))
    bw, bh = int(round((x1 - x0 + 1) * U)), int(round((y1 - y0 + 1) * U))
    w, h = bw + 2 * pad, bh + 2 * pad

    # ---- hladká mapa barev: každá třída se zvětší, rozmaže a vyhraje nejsilnější
    crop = lab[y0:y1 + 1, x0:x1 + 1]
    crop_line = line[y0:y1 + 1, x0:x1 + 1]
    classes = sorted(set(np.unique(crop).tolist()) | {-1})
    LINE_CLS = 1000
    EX = bool(crop_line.any())
    if EX:
        classes.append(LINE_CLS)
    sig = 0.75 * U
    best = np.full((h, w), -1.0, np.float32)
    Lu = np.full((h, w), -1, np.int16)
    for c in classes:
        if c == LINE_CLS:
            m = crop_line
        elif c == -1:
            m = ~((crop >= 0) | crop_line)
        else:
            m = (crop == c) & ~crop_line if EX else (crop == c)
        if not m.any():
            continue
        f = np.zeros((h, w), np.float32)
        f[pad:pad + bh, pad:pad + bw] = cv2.resize(m.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
        if c == -1:  # okolí obrázku je pozadí
            f[:pad, :] = 1; f[pad + bh:, :] = 1; f[:, :pad] = 1; f[:, pad + bw:] = 1
        f = cv2.GaussianBlur(f, (0, 0), sig)
        upd = f > best
        best[upd] = f[upd]
        Lu[upd] = c
    del best
    holeU = np.zeros((h, w), bool)
    if hole.any():
        hc = cv2.resize(hole[y0:y1 + 1, x0:x1 + 1].astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
        holeU[pad:pad + bh, pad:pad + bw] = cv2.GaussianBlur(hc, (0, 0), sig) > 0.5

    def role(c):
        return roles[c] if 0 <= c < len(roles) else "outline"

    def lvl(c):
        if c == LINE_CLS or c < 0:
            return NL if c == LINE_CLS else 0
        r = role(c)
        if r == "board":
            return 1
        if r in ("outline", "wood"):
            return NL
        try:
            return int(max(2, min(NL - 1, int(r))))
        except (TypeError, ValueError):
            return 2

    lineU = Lu == LINE_CLS
    if not EX:  # obrázek bez čar: barva v roli obrys tvoří linky
        for c in classes:
            if c >= 0 and role(c) == "outline":
                lineU |= Lu == c
    fg = Lu >= 0
    levU = np.zeros((h, w), np.int8)
    for c in classes:
        if c == -1:
            continue
        levU[Lu == c] = lvl(c)

    r = max(1, int(round(px(float(opt.get("minFeature", 1.2))) / 2)))
    sepw = float(opt.get("sepLine", 1.5))
    half = max(r, int(round(px(sepw) / 2)))
    min_piece_px = float(opt.get("minPiece", 10)) / (mm * mm)

    # ---- deska a rámeček
    ch, f_ = int(round(px(float(opt.get("chamfer", 4))))), int(round(px(frame)))
    rect = np.ones((h, w), np.uint8)
    if ch > 0:
        tri = np.array([[0, 0], [ch, 0], [0, ch]], np.int32)
        for pts in (tri, [w - 1, 0] + tri * [-1, 1], [0, h - 1] + tri * [1, -1], [w - 1, h - 1] - tri):
            cv2.fillPoly(rect, [np.array(pts, np.int32)], 0)
    rect = rect.astype(bool)
    ring = rect.copy()
    ring[f_:h - f_, f_:w - f_] = False

    # ---- silueta
    sil = drop_small(fg, min_piece_px)
    sil &= rect
    big = holeU & lineU & sil

    # ---- okraj (jen obrázky bez čar)
    rim = np.zeros_like(sil)
    if not EX and float(opt.get("rim", 3)) > 0:
        rim = sil & ~erode(sil, max(r, int(round(px(float(opt.get("rim", 3)))))))

    # ---- dělicí linky mezi barvami, kde chybí čára
    sep = np.zeros_like(sil)
    if sepw > 0:
        cls = Lu.copy()
        cls[lineU] = -5
        cls[big] = -6
        edge = np.zeros_like(sil)
        for dy, dx in ((0, 1), (1, 0)):
            a = cls[: h - dy, : w - dx]
            b = cls[dy:, dx:]
            e = (a != b) & (a >= 0) & (b >= 0)
            edge[: h - dy, : w - dx] |= e
        if EX:
            near_line = dilate(lineU, half + 2)
            edge &= ~near_line
        sep = dilate(edge, half)
        if EX:  # obrys proti pozadí tam, kde čára chybí
            bg = ~fg
            outer = dilate(bg, 1) & fg & ~lineU & ~dilate(lineU, 2)
            sep |= dilate(outer, 2 * half)
        sep = smooth(sep, max(1.0, px(0.25))) & sil
        big &= ~sep

    # ---- barvy v roli Černá (deska): díra ve všech vrstvách až na desku
    board = np.zeros_like(sil)
    for c in classes:
        if c >= 0 and role(c) == "board":
            board |= Lu == c
    if board.any():
        big |= opening(board & sil & ~sep & ~rim, r)
    # drobné díry (tečky mezi tahy) zrušit, zůstanou dřevem
    big = drop_small(big, float(opt.get("minHole", 3)) * 2 / (mm * mm))

    # ---- horní vrstva (síť linek)
    woodU = np.zeros_like(sil)
    for c in classes:
        if c >= 0 and role(c) == "wood":
            woodU |= Lu == c
    if EX:
        top = (lineU & ~big) | (woodU & sil) | sep
    else:
        top = ((lineU & sil & ~big) | (woodU & sil) | rim | sep)
        holes_in = sil & ~big & ~top
        lab_h, n = ndi.label(holes_in)
        if n:
            sizes = np.bincount(lab_h.ravel())
            small = sizes < float(opt.get("minHole", 3)) / (mm * mm)
            small[0] = False
            top |= small[lab_h]
    if opt.get("dropThin"):
        top = opening(top, r) | (rim & top)
    l2 = sil & ~big

    # ---- spojky k rámečku (volitelné)
    n_struts = int(max(0, min(4, opt.get("struts", 0) or 0)))
    if n_struts and f_ > 0:
        sw = max(2 * r, int(round(px(float(opt.get("strutWidth", 4))))))
        core = erode(sil, sw // 2 + 1)
        if not core.any():
            core = sil
        yy, xx = np.where(core)
        pts = {"L": (xx.min(), yy[xx.argmin()]), "R": (xx.max(), yy[xx.argmax()]),
               "T": (xx[yy.argmin()], yy.min()), "B": (xx[yy.argmax()], yy.max())}
        st = np.zeros((h, w), np.uint8)
        for side in "LRBT"[:n_struts]:
            x, y = pts[side]
            q = {"L": (f_ - 2, y), "R": (w - f_ + 1, y), "T": (x, f_ - 2), "B": (x, h - f_ + 1)}[side]
            cv2.line(st, (int(x), int(y)), (int(q[0]), int(q[1])), 1, sw)
        st = st.astype(bool) & rect
        top |= st; l2 |= st; sil |= st; big &= ~st

    # ---- spojit ostrůvky horní vrstvy s hlavní sítí
    n_bridges = 0
    net = top | (ring & rect)
    if opt.get("bridges", True):
        for _ in range(6):
            labn, n = ndi.label(net, structure=np.ones((3, 3)))
            if n <= 1:
                break
            main_id = np.bincount(labn[ring & net].ravel()).argmax() if (ring & net).any() else np.bincount(labn.ravel())[1:].argmax() + 1
            main = labn == main_id
            dist, iy, ix = nearest(main)
            st = np.zeros((h, w), np.uint8)
            added = 0
            for k in range(1, n + 1):
                if k == main_id:
                    continue
                comp = labn == k
                if comp.sum() * mm * mm < 0.3:
                    continue
                dd = np.where(comp, dist, np.inf)
                p = np.unravel_index(np.argmin(dd), dd.shape)
                if not np.isfinite(dd[p]):
                    continue
                q = (iy[p], ix[p])
                cv2.line(st, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, max(1, 2 * half))
                added += 1
            if not added:
                break
            st = st.astype(bool) & sil
            top |= st; l2 |= st; big &= ~st
            net = top | (ring & rect)
            n_bridges += added

    # ---- vrstvy
    M = [rect, l2 | (ring & rect)]
    for k in range(3, NL):
        m = opening((levU >= k) & sil & ~big, r)
        M.append((m | top | (ring & rect)) & M[-1])
    M.append((top | (ring & rect)) & M[-1])

    # kousky: díly vrstvy, které nedrží s rámečkem; drobné zahodit, ostatní značka na vrstvě pod nimi
    pieces = [None] * NL
    n_pieces = 0
    for k in range(2, NL):
        labk, n = ndi.label(M[k], structure=np.ones((3, 3)))
        if n <= 1:
            continue
        sizes = np.bincount(labk.ravel())
        main_id = np.bincount(labk[ring & M[k]].ravel()).argmax()
        small = sizes < min_piece_px
        small[0] = False
        M[k] = M[k] & ~small[labk]
        pc = M[k] & (labk != main_id)
        if pc.any():
            pieces[k] = pc
            n_pieces += int(len(np.unique(labk[pc])))

    # ---- barvy natřených ploch a jejich linky
    painted = np.zeros((h, w), bool)
    for c in classes:
        if c >= 0 and role(c) not in ("outline", "wood", "board"):
            painted |= Lu == c
    if painted.any():
        _, ny, nx = nearest(painted)
        colorIdx = Lu[ny, nx]
    else:
        colorIdx = Lu.copy()
    scale = mm
    paint_groups = [[] for _ in range(NL)]
    for k in range(1, NL - 1):
        vis = M[k] & ~M[k + 1]
        if not vis.any():
            continue
        dist_out = cv2.distanceTransform(M[k].astype(np.uint8), cv2.DIST_L2, 3)
        for c in sorted(set(np.unique(colorIdx[vis]).tolist())):
            if c < 0 or c >= len(cent):
                continue
            reg = vis & (colorIdx == c)
            if reg.sum() * mm * mm < 4:
                continue
            lines = []
            for poly in contours(reg, 1.0, smooth_passes=2, eps=0.8):
                cur = []
                for x, y in poly:
                    xi, yi = min(w - 1, max(0, int(x))), min(h - 1, max(0, int(y)))
                    if dist_out[yi, xi] <= 1.5 + half * 0.5:
                        if len(cur) >= 2:
                            lines.append(np.array(cur) * scale)
                        cur = []
                    else:
                        cur.append((x, y))
                if len(cur) >= 2:
                    lines.append(np.array(cur) * scale)
            if lines:
                col = "#" + "".join(f"{int(v):02x}" for v in cent[c])
                paint_groups[k].append((col, lines))

    # ---- SVG
    wmm, hmm = w * mm, h * mm
    labels = []
    for k in range(NL):
        if k == 0:
            labels.append("1 Deska - natřít černě")
        elif k == NL - 1:
            labels.append(f"{k + 1} Čisté dřevo")
        else:
            labels.append(f"{k + 1} Barvy - natřít podle náhledu")
    sheets = []
    cut_px = []
    for k in range(NL):
        mat = M[k] if k == 0 else (M[k] | (ring & rect))
        cut = contours(mat, scale)
        mark = contours(M[1], scale) if k == 0 else (contours(pieces[k + 1], scale) if k + 1 < NL and pieces[k + 1] is not None else [])
        sheets.append({"label": labels[k], "cut": cut, "mark": mark, "paint": paint_groups[k]})
        cut_px.append(cut)
    svg = combined_svg(wmm, hmm, sheets)

    # ---- náhledy vrstev (2x původní rozlišení)
    ps = max(1, int(math.ceil(max(w, h) / 1600)))   # náhled nejvýš ~1600 px, ať se prohlížeč neseká
    pw, ph = w // ps, h // ps
    previews = []
    for k in range(NL):
        mat = M[k] if k == 0 else (M[k] | (ring & rect))
        img = np.zeros((h, w, 4), np.uint8)
        if k == 0:
            img[mat] = (30, 30, 34, 255)
            img[dilate(M[1], 1) & ~erode(M[1], 1) & mat] = (0, 190, 205, 255)
        else:
            wood = (232, 204, 160, 255) if k == NL - 1 else (216 + 4 * k, 184 + 4 * k, 138 + 4 * k, 255)
            img[mat] = wood
            if k < NL - 1:
                vis = M[k] & ~M[k + 1]
                cols = np.array([list(c) + [255] for c in cent], np.uint8)
                ci = np.clip(colorIdx, 0, len(cent) - 1)
                img[vis] = cols[ci[vis]]
        im = Image.fromarray(img, "RGBA").resize((pw, ph), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=False, compress_level=3)
        previews.append(base64.b64encode(buf.getvalue()).decode())
    pscale = mm * ps  # mm na pixel náhledu
    cut_preview = [path_d([p / pscale for p in polys]) for polys in cut_px]

    holes_n = int(ndi.label(big)[1])
    sil_parts = int(ndi.label(l2, structure=np.ones((3, 3)))[1])
    joined = int(ndi.label(M[NL - 1] | (ring & rect), structure=np.ones((3, 3)))[1]) == 1
    return {
        "svg": svg,
        "layers": previews,
        "lw": int(pw), "lh": int(ph),
        "cutPx": cut_preview,
        "mm": float(pscale),
        "stats": {"wmm": float(wmm), "hmm": float(hmm), "silParts": int(sil_parts), "holes": int(holes_n), "pieces": int(n_pieces),
                  "layers": int(NL), "bridges": int(n_bridges), "frameJoined": bool(joined), "engine": "server", "upscale": round(float(U), 1)},
    }


def combined_svg(wmm, hmm, sheets):
    gap, lab, cols = 10.0, 14.0, 2
    rows = math.ceil(len(sheets) / cols)
    W = cols * wmm + (cols - 1) * gap
    H = rows * (hmm + lab) + (rows - 1) * gap
    out = [f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}mm" height="{H:.2f}mm" viewBox="0 0 {W:.3f} {H:.3f}"><title>vrstvy pro laser</title>']
    for i, sh in enumerate(sheets):
        ox = (i % cols) * (wmm + gap)
        oy = (i // cols) * (hmm + lab + gap)
        off = np.array([ox, oy + lab])
        out.append(f'<g id="vrstva{i + 1}">')
        out.append(f'<text x="{ox + wmm / 2:.2f}" y="{oy + lab * 0.72:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="9" fill="#ff0000">{sh["label"]}</text>')
        if sh["cut"]:
            out.append(f'<path style="stroke:{CUT};stroke-width:0.05;fill:none" d="{path_d([p + off for p in sh["cut"]])}"/>')
        if sh["mark"]:
            out.append(f'<path style="stroke:{MARK};stroke-width:0.05;fill:none" d="{path_d([p + off for p in sh["mark"]])}"/>')
        for col, lines in sh["paint"]:
            out.append(f'<path style="stroke:{col};stroke-width:0.05;fill:none" d="{open_lines_d([l + off for l in lines])}"/>')
        out.append("</g>")
    out.append("</svg>\n")
    return "\n".join(out)
