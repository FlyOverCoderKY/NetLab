import React, { useEffect, useRef, useState } from "react";
import { useAppStore } from "../../state/store";
import { getTrainerClient } from "../../worker/client";

type Box = { x: number; y: number; w: number; h: number };

const EXAMPLES = [
  { name: "Hello World", text: "HELLO WORLD 123" },
  { name: "Quick Fox", text: "THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS" },
  { name: "Digits", text: "0123456789 9876543210" },
];

function drawTextToCanvas(
  text: string,
  family: string,
  fontPx: number,
  width = 640,
): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = 200;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#000";
  ctx.font = `${Math.max(8, Math.min(64, fontPx | 0))}px ${family}, sans-serif`;
  ctx.textBaseline = "top";
  const words = text.split(/\s+/);
  let x = 10,
    y = 10;
  for (const w of words) {
    const m = ctx.measureText(w + " ");
    if (x + m.width > c.width - 10) {
      x = 10;
      y += 36;
    }
    ctx.fillText(w + " ", x, y);
    x += m.width;
  }
  return c;
}

// Adaptive thresholding (Sauvola-like) using box integral images
function binarizeAdaptive(
  img: ImageData,
  windowRadius = 8,
  k = 0.2,
): Uint8ClampedArray {
  const w = img.width;
  const h = img.height;
  const N = w * h;
  const gray = new Float32Array(N);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    const r = img.data[i];
    const g = img.data[i + 1];
    const b = img.data[i + 2];
    gray[j] = (r + g + b) / 3;
  }
  const integral = new Float64Array((w + 1) * (h + 1));
  const integralSq = new Float64Array((w + 1) * (h + 1));
  const idxI = (x: number, y: number) => y * (w + 1) + x;
  for (let y = 1; y <= h; y++) {
    let rowSum = 0;
    let rowSumSq = 0;
    for (let x = 1; x <= w; x++) {
      const v = gray[(y - 1) * w + (x - 1)];
      rowSum += v;
      rowSumSq += v * v;
      integral[idxI(x, y)] = integral[idxI(x, y - 1)] + rowSum;
      integralSq[idxI(x, y)] = integralSq[idxI(x, y - 1)] + rowSumSq;
    }
  }
  const out = new Uint8ClampedArray(N);
  const r = Math.max(1, windowRadius);
  const kLocal = k;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r);
    const y1 = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r);
      const x1 = Math.min(w - 1, x + r);
      const A = idxI(x0, y0);
      const B = idxI(x1 + 1, y0);
      const C = idxI(x0, y1 + 1);
      const D = idxI(x1 + 1, y1 + 1);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const sum = integral[D] - integral[B] - integral[C] + integral[A];
      const sumSq =
        integralSq[D] - integralSq[B] - integralSq[C] + integralSq[A];
      const mean = sum / area;
      const variance = Math.max(0, sumSq / area - mean * mean);
      const std = Math.sqrt(variance);
      const threshold = mean * (1 + kLocal * (std / 128 - 1));
      const v = gray[y * w + x];
      out[y * w + x] = v < threshold ? 1 : 0;
    }
  }
  return out;
}

function erode(bin: Uint8ClampedArray, w: number, h: number, iters = 1) {
  let src = bin;
  for (let t = 0; t < iters; t++) {
    const dst = new Uint8ClampedArray(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (src[idx] === 0) continue;
        const left = x > 0 ? src[idx - 1] : 0;
        const right = x + 1 < w ? src[idx + 1] : 0;
        const up = y > 0 ? src[idx - w] : 0;
        const down = y + 1 < h ? src[idx + w] : 0;
        dst[idx] = left && right && up && down ? 1 : 0;
      }
    }
    src = dst;
  }
  return src;
}

function cropTo28x28(src: ImageData, b: Box, contentTarget = 20): Float32Array {
  const out = new Float32Array(28 * 28);
  // Light trim of outer whitespace to reduce neighbor bleed
  let bx = b.x,
    by = b.y,
    bw = b.w,
    bh = b.h;
  const colInk = (cx: number): number => {
    let s = 0;
    for (let y = by; y < by + bh; y++) {
      const idx = (y * src.width + cx) * 4;
      const r = src.data[idx];
      const g = src.data[idx + 1];
      const bb = src.data[idx + 2];
      s += 255 - (r + g + bb) / 3;
    }
    return s;
  };
  const rowInk = (ry: number): number => {
    let s = 0;
    const off = ry * src.width;
    for (let x = bx; x < bx + bw; x++) {
      const idx = (off + x) * 4;
      const r = src.data[idx];
      const g = src.data[idx + 1];
      const bb = src.data[idx + 2];
      s += 255 - (r + g + bb) / 3;
    }
    return s;
  };
  const colThresh = Math.max(1, bh * 6); // small amount of ink
  const rowThresh = Math.max(1, bw * 6);
  let trims = 0;
  while (bw > 8 && colInk(bx) <= colThresh && trims < 3) {
    bx++;
    bw--;
    trims++;
  }
  trims = 0;
  while (bw > 8 && colInk(bx + bw - 1) <= colThresh && trims < 3) {
    bw--;
    trims++;
  }
  trims = 0;
  while (bh > 8 && rowInk(by) <= rowThresh && trims < 2) {
    by++;
    bh--;
    trims++;
  }
  trims = 0;
  while (bh > 8 && rowInk(by + bh - 1) <= rowThresh && trims < 2) {
    bh--;
    trims++;
  }
  const content = Math.max(16, Math.min(24, contentTarget | 0));
  const scale = Math.min(content / bw, content / bh);
  const tw = Math.max(1, Math.floor(bw * scale));
  const th = Math.max(1, Math.floor(bh * scale));
  // Compute center of mass to center ink
  let sumX = 0,
    sumY = 0,
    sum = 0;
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const sx = bx + x;
      const sy = by + y;
      const idx = (sy * src.width + sx) * 4;
      const r = src.data[idx];
      const g = src.data[idx + 1];
      const bb = src.data[idx + 2];
      const v = 255 - (r + g + bb) / 3;
      sumX += x * v;
      sumY += y * v;
      sum += v;
    }
  }
  const cx = sum > 0 ? sumX / sum : bw / 2;
  const cy = sum > 0 ? sumY / sum : bh / 2;
  const ox = Math.floor(14 - cx * scale);
  const oy = Math.floor(14 - cy * scale);
  // Rasterize via canvas to mimic generator anti-aliasing behavior
  const off = document.createElement("canvas");
  off.width = 28;
  off.height = 28;
  const ctx = off.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, 28, 28);
  // Put source image into a temporary canvas to use drawImage source rect
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = src.width;
  srcCanvas.height = src.height;
  const sctx = srcCanvas.getContext("2d")!;
  const tmp = sctx.createImageData(src.width, src.height);
  tmp.data.set(src.data);
  sctx.putImageData(tmp, 0, 0);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, 28, 28);
  ctx.drawImage(
    srcCanvas,
    bx,
    by,
    bw,
    bh,
    Math.max(0, ox),
    Math.max(0, oy),
    tw,
    th,
  );
  const img2 = ctx.getImageData(0, 0, 28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const idx = i * 4;
    const r = img2.data[idx];
    const g = img2.data[idx + 1];
    const bch = img2.data[idx + 2];
    out[i] = (r + g + bch) / (3 * 255);
  }
  return out;
}

function estimateSkewAngle(
  bin: Uint8ClampedArray,
  w: number,
  h: number,
): number {
  // Downscale for speed
  const targetW = Math.min(256, w);
  const scale = targetW / w;
  const targetH = Math.max(1, Math.floor(h * scale));
  const small = new Uint8Array(targetW * targetH);
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const sx = Math.min(w - 1, Math.floor(x / scale));
      const sy = Math.min(h - 1, Math.floor(y / scale));
      small[y * targetW + x] = bin[sy * w + sx];
    }
  }
  let bestAngle = 0;
  let bestScore = -Infinity;
  const angles = [] as number[];
  for (let a = -8; a <= 8; a += 1) angles.push(a * (Math.PI / 180));
  const cx = targetW / 2;
  const cy = targetH / 2;
  for (const ang of angles) {
    const s = Math.sin(ang);
    const c = Math.cos(ang);
    const hist = new Float32Array(targetH);
    for (let y = 0; y < targetH; y++) {
      for (let x = 0; x < targetW; x++) {
        if (small[y * targetW + x] === 0) continue;
        // rotate point around center and accumulate along y-axis
        const yr = -s * (x - cx) + c * (y - cy) + cy;
        const yi = Math.round(yr);
        if (yi >= 0 && yi < targetH) hist[yi] += 1;
      }
    }
    // Score: variance of histogram (sharper lines -> higher variance)
    let mean = 0;
    for (let i = 0; i < targetH; i++) mean += hist[i];
    mean /= targetH || 1;
    let varSum = 0;
    for (let i = 0; i < targetH; i++) {
      const d = hist[i] - mean;
      varSum += d * d;
    }
    if (varSum > bestScore) {
      bestScore = varSum;
      bestAngle = ang;
    }
  }
  return bestAngle * (180 / Math.PI);
}

function rotateCanvas(
  src: HTMLCanvasElement,
  angleDeg: number,
): HTMLCanvasElement {
  const ang = (angleDeg / 180) * Math.PI;
  if (Math.abs(angleDeg) < 0.1) return src;
  const s = Math.sin(ang);
  const c = Math.cos(ang);
  const w = src.width;
  const h = src.height;
  const newW = Math.ceil(Math.abs(w * c) + Math.abs(h * s));
  const newH = Math.ceil(Math.abs(w * s) + Math.abs(h * c));
  const out = document.createElement("canvas");
  out.width = newW;
  out.height = newH;
  const ctx = out.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, newW, newH);
  ctx.translate(newW / 2, newH / 2);
  ctx.rotate(ang);
  ctx.drawImage(src, -w / 2, -h / 2);
  return out;
}

// Line-first segmentation using horizontal/vertical projections
function segmentByProjections(
  bin: Uint8ClampedArray,
  w: number,
  h: number,
): Box[] {
  const boxes: Box[] = [];
  const pad = (b: Box, p = 1): Box => ({
    x: Math.max(0, b.x - p),
    y: Math.max(0, b.y - p),
    w: Math.min(w - Math.max(0, b.x - p), b.w + 2 * p),
    h: Math.min(h - Math.max(0, b.y - p), b.h + 2 * p),
  });

  // Try splitting a suspiciously wide box using connected components inside its region
  const splitByCC = (bb: Box): Box[] => {
    const sx0 = bb.x;
    const sy0 = bb.y;
    const sw = bb.w;
    const sh = bb.h;
    const sub = new Uint8ClampedArray(sw * sh);
    for (let yy = 0; yy < sh; yy++) {
      const row = (sy0 + yy) * w;
      for (let xx = 0; xx < sw; xx++) sub[yy * sw + xx] = bin[row + (sx0 + xx)];
    }
    const cc = (srcBin: Uint8ClampedArray): Box[] => {
      const seen = new Uint8Array(srcBin.length);
      const comps: Box[] = [];
      const qx: number[] = [];
      const qy: number[] = [];
      const push = (x: number, y: number) => {
        qx.push(x);
        qy.push(y);
      };
      const pop = () => ({ x: qx.pop()!, y: qy.pop()! });
      const inside = (x: number, y: number) =>
        x >= 0 && y >= 0 && x < sw && y < sh;
      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const idx = y * sw + x;
          if (srcBin[idx] === 1 && !seen[idx]) {
            let minx = x,
              maxx = x,
              miny = y,
              maxy = y,
              area = 0;
            push(x, y);
            seen[idx] = 1;
            while (qx.length) {
              const { x: cx, y: cy } = pop();
              area++;
              minx = Math.min(minx, cx);
              miny = Math.min(miny, cy);
              maxx = Math.max(maxx, cx);
              maxy = Math.max(maxy, cy);
              const neigh = [
                [cx + 1, cy],
                [cx - 1, cy],
                [cx, cy + 1],
                [cx, cy - 1],
              ];
              for (const [nx, ny] of neigh) {
                if (!inside(nx, ny)) continue;
                const nidx = ny * sw + nx;
                if (srcBin[nidx] === 1 && !seen[nidx]) {
                  seen[nidx] = 1;
                  push(nx, ny);
                }
              }
            }
            if (area >= Math.max(5, Math.floor(sw * sh * 0.01))) {
              comps.push({
                x: sx0 + minx,
                y: sy0 + miny,
                w: maxx - minx + 1,
                h: maxy - miny + 1,
              });
            }
          }
        }
      }
      comps.sort((a, b) => a.x - b.x);
      return comps;
    };
    let comps = cc(sub);
    if (comps.length < 2) {
      // Try a light erosion inside this region to break thin bridges (e.g., I next to C)
      const eroded2 = erode(sub, sw, sh, 2);
      comps = cc(eroded2);
    }
    if (comps.length < 2) {
      // Try slightly stronger erosion
      const eroded3 = erode(sub, sw, sh, 3);
      comps = cc(eroded3);
    }
    if (comps.length < 2) {
      // Valley whitening: zero out columns that are near-empty to force CC separation
      const cols = new Uint16Array(sw);
      for (let xx = 0; xx < sw; xx++) {
        let s = 0;
        for (let yy = 0; yy < sh; yy++) s += sub[yy * sw + xx];
        cols[xx] = s;
      }
      const valley = Math.max(1, Math.floor(sh * 0.06));
      const whitened = new Uint8ClampedArray(sub);
      let changed = false;
      for (let xx = 1; xx < sw - 1; xx++) {
        if (
          cols[xx] <= valley &&
          (cols[xx - 1] <= valley || cols[xx + 1] <= valley)
        ) {
          for (let yy = 0; yy < sh; yy++) whitened[yy * sw + xx] = 0;
          changed = true;
        }
      }
      if (changed) comps = cc(whitened);
    }
    return comps.length >= 2 ? comps : [bb];
  };
  // Horizontal projection to find line bands
  const rows = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let s = 0;
    const off = y * w;
    for (let x = 0; x < w; x++) s += bin[off + x];
    rows[y] = s;
  }
  // Smooth
  for (let y = 1; y < h - 1; y++)
    rows[y] = (rows[y - 1] + rows[y] + rows[y + 1]) / 3;
  const maxRow = rows.reduce((a, b) => (b > a ? b : a), 0);
  const rowThresh = Math.max(1, maxRow * 0.15);
  let y = 0;
  while (y < h) {
    // skip whitespace rows
    while (y < h && rows[y] <= rowThresh) y++;
    if (y >= h) break;
    const y0 = y;
    while (y < h && rows[y] > rowThresh) y++;
    const y1 = y - 1;
    const lineH = y1 - y0 + 1;
    if (lineH < 8) continue;
    // Vertical projection within line to split into glyph boxes
    const cols = new Float32Array(w);
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let yy = y0; yy <= y1; yy++) s += bin[yy * w + x];
      cols[x] = s;
    }
    // Smooth
    for (let x = 1; x < w - 1; x++)
      cols[x] = (cols[x - 1] + cols[x] + cols[x + 1]) / 3;
    const valley = Math.max(1, Math.floor(lineH * 0.12));
    const minGlyphW = Math.max(6, Math.floor(lineH * 0.35));
    const maxGlyphW = Math.max(minGlyphW + 1, Math.floor(lineH * 1.2));
    let start = 0;
    // skip leading whitespace
    while (start < w && cols[start] <= valley) start++;
    let x = start;
    const lineBoxes: Box[] = [];
    const pushSeg = (s: number, e: number) => {
      const gw = e - s;
      if (gw >= Math.max(4, Math.floor(lineH * 0.3))) {
        lineBoxes.push(pad({ x: s, y: y0, w: gw, h: lineH }));
      }
    };
    while (x < w) {
      // find next valley run
      const x2 = x + minGlyphW;
      if (x2 >= w) break;
      // search for a run of near-zero columns that indicates whitespace
      let cutAt = -1;
      for (let j = x2; j < Math.min(w - 1, x + maxGlyphW); j++) {
        const isValley = cols[j] <= valley && cols[j + 1] <= valley;
        if (isValley) {
          cutAt = j;
          break;
        }
      }
      if (cutAt > 0) {
        pushSeg(x, cutAt);
        // advance to next non-valley
        let nx = cutAt + 1;
        while (nx < w && cols[nx] <= valley) nx++;
        x = nx;
      } else {
        // no good valley; extend and force a cut if extremely wide
        if (x + maxGlyphW >= w) {
          pushSeg(x, w);
          break;
        }
        // Force cut at local minimum in the search window
        let bestJ = x + minGlyphW;
        let bestV = cols[bestJ];
        for (let j = x + minGlyphW; j < Math.min(w, x + maxGlyphW); j++) {
          if (cols[j] < bestV) {
            bestV = cols[j];
            bestJ = j;
          }
        }
        pushSeg(x, bestJ);
        x = Math.min(w, bestJ + 1);
        while (x < w && cols[x] <= valley) x++;
      }
    }
    // Refine: if any box is ~two glyphs wide (e.g., IC), try a micro-split at best local minimum
    if (lineBoxes.length) {
      const widths = lineBoxes.map((b2) => b2.w).sort((a, b3) => a - b3);
      const medianW = widths[Math.floor(widths.length / 2)] || minGlyphW;
      const refined: Box[] = [];
      for (const bb of lineBoxes) {
        if (bb.w > 1.5 * medianW && bb.w < 3.0 * medianW) {
          const sm = new Float32Array(bb.w);
          for (let i = 0; i < bb.w; i++) {
            let s = 0;
            for (let yy = bb.y; yy < bb.y + bb.h; yy++)
              s += bin[yy * w + (bb.x + i)];
            sm[i] = s;
          }
          for (let i = 1; i < sm.length - 1; i++)
            sm[i] = (sm[i - 1] + sm[i] + sm[i + 1]) / 3;
          // Prefer hard whitespace runs (>=2 columns nearly zero) inside the center band
          let runStart = -1,
            runLen = 0,
            bestRunMid = -1;
          for (
            let i = Math.floor(bb.w * 0.3);
            i < Math.floor(bb.w * 0.7);
            i++
          ) {
            const isWhite = sm[i] <= Math.max(1, Math.floor(lineH * 0.06));
            if (isWhite) {
              if (runLen === 0) runStart = i;
              runLen++;
            } else if (runLen > 0) {
              if (runLen >= 2)
                bestRunMid = Math.floor((runStart + (i - 1)) / 2);
              runLen = 0;
            }
          }
          if (runLen >= 2 && bestRunMid < 0)
            bestRunMid = Math.floor(
              (runStart + (Math.floor(bb.w * 0.7) - 1)) / 2,
            );
          let bestX = Math.floor(bb.w / 2);
          let bestV = sm[bestX];
          if (bestRunMid >= 0) {
            bestX = bestRunMid;
            bestV = sm[bestRunMid];
          } else {
            for (
              let i = Math.floor(bb.w * 0.35);
              i < Math.floor(bb.w * 0.65);
              i++
            ) {
              if (sm[i] < bestV) {
                bestV = sm[i];
                bestX = i;
              }
            }
          }
          const leftPeak = Math.max(
            sm[Math.max(0, bestX - 3)],
            sm[Math.max(0, bestX - 6)] || sm[Math.max(0, bestX - 3)],
          );
          const rightPeak = Math.max(
            sm[Math.min(sm.length - 1, bestX + 3)],
            sm[Math.min(sm.length - 1, bestX + 6)] ||
              sm[Math.min(sm.length - 1, bestX + 3)],
          );
          const deepEnough =
            bestV <= Math.max(1, lineH * 0.08) ||
            bestV <= 0.35 * Math.min(leftPeak, rightPeak);
          const lW = bestX;
          const rW = bb.w - bestX;
          if (
            deepEnough &&
            lW >= Math.max(4, Math.floor(medianW * 0.3)) &&
            rW >= Math.max(5, Math.floor(medianW * 0.6))
          ) {
            refined.push(pad({ x: bb.x, y: bb.y, w: bestX, h: bb.h }));
            refined.push(
              pad({ x: bb.x + bestX, y: bb.y, w: bb.w - bestX, h: bb.h }),
            );
            continue;
          }
        }
        // Fallback: connected-components split inside the wide region (helps with thin I + C)
        if (bb.w > 1.8 * medianW) {
          const cc = splitByCC(bb);
          if (cc.length >= 2) {
            refined.push(...cc.map((bseg) => pad(bseg)));
            continue;
          }
          // Mass-based fallback: cut where ~1/3 of ink mass accumulates, biased to a nearby low column
          const sm2 = new Float32Array(bb.w);
          let total = 0;
          for (let i = 0; i < bb.w; i++) {
            let s = 0;
            for (let yy = bb.y; yy < bb.y + bb.h; yy++)
              s += bin[yy * w + (bb.x + i)];
            sm2[i] = s;
            total += s;
          }
          const target = total * 0.35;
          let acc = 0;
          let idx = Math.floor(bb.w * 0.4);
          for (
            let i = Math.floor(bb.w * 0.25);
            i < Math.floor(bb.w * 0.7);
            i++
          ) {
            acc += sm2[i];
            if (acc >= target) {
              idx = i;
              break;
            }
          }
          // refine idx to a nearby local minimum within ±3
          let bestI = idx,
            bestV = sm2[idx];
          for (
            let j = Math.max(1, idx - 3);
            j <= Math.min(bb.w - 2, idx + 3);
            j++
          ) {
            if (sm2[j] < bestV) {
              bestV = sm2[j];
              bestI = j;
            }
          }
          const lW = bestI;
          const rW = bb.w - bestI;
          if (
            lW >= Math.max(5, Math.floor(medianW * 0.4)) &&
            rW >= Math.max(5, Math.floor(medianW * 0.4))
          ) {
            refined.push(pad({ x: bb.x, y: bb.y, w: bestI, h: bb.h }));
            refined.push(
              pad({ x: bb.x + bestI, y: bb.y, w: bb.w - bestI, h: bb.h }),
            );
            continue;
          }
        }
        refined.push(pad(bb));
      }
      boxes.push(...refined);
    }
  }
  return boxes.sort((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y));
}

function insertSpaces(sorted: Box[]): Box[] {
  if (sorted.length < 3) return sorted;
  // Compute median gap within approximate lines (cluster by y)
  const groups: Box[][] = [];
  let current: Box[] = [];
  const lineH = Math.max(
    10,
    Math.round(sorted.reduce((a, b) => a + b.h, 0) / sorted.length),
  );
  for (const b of sorted) {
    if (current.length === 0 || Math.abs(b.y - current[0].y) < lineH)
      current.push(b);
    else {
      groups.push(current);
      current = [b];
    }
  }
  if (current.length) groups.push(current);
  const spaced: Box[] = [];
  for (const g of groups) {
    g.sort((a, b) => a.x - b.x);
    const gaps: number[] = [];
    for (let i = 1; i < g.length; i++)
      gaps.push(g[i].x - (g[i - 1].x + g[i - 1].w));
    const sortedG = gaps.slice().sort((a, b) => a - b);
    const median = sortedG.length ? sortedG[Math.floor(sortedG.length / 2)] : 0;
    for (let i = 0; i < g.length; i++) {
      spaced.push(g[i]);
      if (i < g.length - 1) {
        const gap = g[i + 1].x - (g[i].x + g[i].w);
        if (median > 0 && gap > 1.75 * median) {
          // insert a sentinel box indicating a space
          spaced.push({
            x: g[i].x + g[i].w + gap / 2,
            y: g[i].y,
            w: 0,
            h: g[i].h,
          });
        }
      }
    }
  }
  return spaced;
}

const OCRPanel: React.FC = () => {
  const dataset = useAppStore((s) => s.dataset);
  const [image, setImage] = useState<HTMLCanvasElement | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [text, setText] = useState<string>("");
  const uploadRef = useRef<HTMLInputElement | null>(null);
  const [procCrops, setProcCrops] = useState<Float32Array[]>([]);
  const setOcrSamples = useAppStore((s) => s.setOcrSamples);

  const choose = (t: string) => {
    const base = drawTextToCanvas(t, dataset.fontFamily, dataset.fontSize);
    const ctx0 = base.getContext("2d")!;
    const img0 = ctx0.getImageData(0, 0, base.width, base.height);
    const bin0 = binarizeAdaptive(img0, 10, 0.2);
    const angle = estimateSkewAngle(bin0, base.width, base.height);
    const c = rotateCanvas(base, angle);
    setImage(c);
    const ctx = c.getContext("2d")!;
    const img = ctx.getImageData(0, 0, c.width, c.height);
    const bin = binarizeAdaptive(img, 10, 0.2);
    const comps = segmentByProjections(bin, c.width, c.height);
    setBoxes(comps);
  };

  const run = async () => {
    if (!image) return;
    const ctx = image.getContext("2d")!;
    const img = ctx.getImageData(0, 0, image.width, image.height);
    const seq = insertSpaces(boxes);
    // Batch crops
    const crops: Float32Array[] = [];
    seq.forEach((b) => {
      if (b.w !== 0) {
        // Fit content into ~training target size (OCR preset ~0.8 scale ~ 20px)
        const crop = cropTo28x28(img, b, 20);
        // Match training polarity: if dataset.invert is true, invert here
        if (dataset.invert) {
          for (let i = 0; i < crop.length; i++) crop[i] = 1 - crop[i];
        }
        crops.push(crop);
      }
    });
    const n = crops.length;
    const batched = new Float32Array(n * 28 * 28);
    for (let i = 0; i < n; i++) batched.set(crops[i], i * 28 * 28);
    const c = getTrainerClient();
    const result = await new Promise<{ probs: Float32Array; n: number }>(
      (resolve) => {
        const off = c.on("prediction-batch", (p) => {
          resolve(p);
          off();
        });
        c.predictBatch(batched, n);
      },
    );
    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let out = "";
    let pi = 0;
    for (const b of seq) {
      if (b.w === 0) {
        out += " ";
        continue;
      }
      const probs = result.probs.slice(pi * 36, (pi + 1) * 36);
      pi++;
      const arr = Array.from(probs as unknown as number[]);
      // Bigram-biased decoding: adjust with simple language prior
      const bigram = (prev: string, nextIdx: number) => {
        // Heuristic: discourage digit after digit when width/height suggests letter, encourage vowels after consonants
        if (!prev) return 0;
        const next = chars[nextIdx] ?? "";
        const vowels = new Set(["A", "E", "I", "O", "U"]);
        let bonus = 0;
        if (/[A-Z]/.test(prev) && !vowels.has(prev) && vowels.has(next))
          bonus += 0.02;
        if (/\d/.test(prev) && /\d/.test(next)) bonus -= 0.02;
        return bonus;
      };
      const prevChar = out.slice(-1);
      let best = -Infinity;
      let bi = 0;
      for (let i = 0; i < arr.length; i++) {
        const score = arr[i] + bigram(prevChar, i);
        if (score > best) {
          best = score;
          bi = i;
        }
      }
      const ar = b.w / Math.max(1, b.h);
      if (ar > 0.65 && (bi === 8 || bi === 1)) {
        const pairs = arr.map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p);
        for (const cand of pairs) {
          if (cand.i !== 8 && cand.i !== 1) {
            bi = cand.i;
            break;
          }
        }
      }
      if (Math.abs(arr[14] - arr[24]) < 0.05 && ar > 0.8 && ar < 1.2) bi = 14;
      out += chars[bi] ?? "?";
    }
    setText(out);
  };

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      const base = document.createElement("canvas");
      base.width = Math.min(img.width, 1024);
      base.height = Math.floor((img.height / img.width) * base.width);
      const ctxb = base.getContext("2d")!;
      ctxb.fillStyle = "#fff";
      ctxb.fillRect(0, 0, base.width, base.height);
      ctxb.drawImage(img, 0, 0, base.width, base.height);
      const id0 = ctxb.getImageData(0, 0, base.width, base.height);
      const bin0 = binarizeAdaptive(id0, 12, 0.2);
      const angle = estimateSkewAngle(bin0, base.width, base.height);
      const c = rotateCanvas(base, angle);
      setImage(c);
      const ctx = c.getContext("2d")!;
      const id = ctx.getImageData(0, 0, c.width, c.height);
      const bin = binarizeAdaptive(id, 12, 0.2);
      const comps = segmentByProjections(bin, c.width, c.height);
      setBoxes(comps);
    };
    img.src = URL.createObjectURL(f);
  };

  // Build previews whenever image/boxes change
  useEffect(() => {
    if (!image || boxes.length === 0) {
      setProcCrops([]);
      return;
    }
    try {
      const ctx = image.getContext("2d")!;
      const img = ctx.getImageData(0, 0, image.width, image.height);
      const out: Float32Array[] = [];
      for (const b of boxes) {
        if (b.w === 0) continue;
        const crop = cropTo28x28(img, b, 20);
        if (dataset.invert)
          for (let i = 0; i < crop.length; i++) crop[i] = 1 - crop[i];
        out.push(crop);
        if (out.length >= 120) break; // cap for UI
      }
      setProcCrops(out);
      setOcrSamples(out);
    } catch {
      setProcCrops([]);
      setOcrSamples([]);
    }
  }, [image, boxes, dataset.invert, setOcrSamples]);

  return (
    <section style={{ padding: "1rem" }}>
      <h3>OCR</h3>
      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              marginBottom: 8,
            }}
          >
            {EXAMPLES.map((e) => (
              <button key={e.name} onClick={() => choose(e.text)}>
                {e.name}
              </button>
            ))}
            <button onClick={() => uploadRef.current?.click()}>
              Upload image…
            </button>
            <input
              ref={uploadRef}
              type="file"
              accept="image/*"
              onChange={onUpload}
              style={{ display: "none" }}
            />
            <button onClick={run} disabled={!image || boxes.length === 0}>
              Run OCR
            </button>
          </div>
          <div
            style={{
              position: "relative",
              border: "1px solid var(--color-border)",
              width: image?.width ?? 0,
              height: image?.height ?? 0,
            }}
          >
            {image ? (
              <canvas
                width={image.width}
                height={image.height}
                ref={(el) => {
                  if (!el || !image) return;
                  const ctx = el.getContext("2d")!;
                  ctx.drawImage(image, 0, 0);
                  ctx.strokeStyle = "rgba(255,0,0,0.6)";
                  boxes.forEach((b) => ctx.strokeRect(b.x, b.y, b.w, b.h));
                }}
              />
            ) : (
              <div
                style={{ padding: 20, color: "var(--color-foreground-subtle)" }}
              >
                Choose an example or upload an image.
              </div>
            )}
          </div>
        </div>
        <div style={{ minWidth: 260 }}>
          <h4>Recognized</h4>
          <pre style={{ whiteSpace: "pre-wrap" }}>{text || "—"}</pre>
          {image && boxes.length ? (
            <div style={{ marginTop: 12 }}>
              <h4>Segments preview</h4>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    Raw (cropped)
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(6, 1fr)",
                      gap: 6,
                    }}
                  >
                    {boxes
                      .filter((b) => b.w !== 0)
                      .slice(0, 36)
                      .map((b, i) => (
                        <RawCrop
                          key={`${b.x}-${b.y}-${i}`}
                          image={image}
                          box={b}
                        />
                      ))}
                  </div>
                </div>
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    Model inputs (28×28)
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(6, 1fr)",
                      gap: 6,
                    }}
                  >
                    {procCrops.slice(0, 36).map((arr, i) => (
                      <Canvas28 key={i} data={arr} />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
};

const Canvas28: React.FC<{ data: Float32Array }> = ({ data }) => {
  const ref = React.useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const size = 56;
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 28 * 28; i++) {
      const v = Math.max(0, Math.min(1, data[i])) * 255;
      img.data[i * 4 + 0] = v;
      img.data[i * 4 + 1] = v;
      img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    const off = document.createElement("canvas");
    off.width = 28;
    off.height = 28;
    const octx = off.getContext("2d")!;
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(off, 0, 0, 28, 28, 0, 0, size, size);
    ctx.strokeStyle = "var(--color-border)";
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  }, [data]);
  return <canvas ref={ref} aria-label="crop-28" />;
};

const RawCrop: React.FC<{ image: HTMLCanvasElement; box: Box }> = ({
  image,
  box,
}) => {
  const ref = React.useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const size = 56;
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, size, size);
    const scale = Math.min(
      size / Math.max(1, box.w),
      size / Math.max(1, box.h),
    );
    const dw = Math.max(1, Math.floor(box.w * scale));
    const dh = Math.max(1, Math.floor(box.h * scale));
    const dx = Math.floor((size - dw) / 2);
    const dy = Math.floor((size - dh) / 2);
    ctx.drawImage(image, box.x, box.y, box.w, box.h, dx, dy, dw, dh);
    ctx.strokeStyle = "var(--color-border)";
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  }, [image, box]);
  return <canvas ref={ref} aria-label="crop-raw" />;
};

export default OCRPanel;
