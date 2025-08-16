import React, { useRef, useState } from "react";
import { getTrainerClient } from "../../worker/client";

type Box = { x: number; y: number; w: number; h: number };

const EXAMPLES = [
  { name: "Hello World", text: "HELLO WORLD 123" },
  { name: "Quick Fox", text: "THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS" },
  { name: "Digits", text: "0123456789 9876543210" },
];

function drawTextToCanvas(text: string, width = 640): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = 200;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#000";
  ctx.font = "28px sans-serif";
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

function binarize(img: ImageData): Uint8ClampedArray {
  const out = new Uint8ClampedArray(img.width * img.height);
  for (let i = 0; i < img.data.length; i += 4) {
    const r = img.data[i],
      g = img.data[i + 1],
      b = img.data[i + 2];
    const gray = (r + g + b) / 3;
    out[i / 4] = gray < 180 ? 1 : 0;
  }
  return out;
}

function findComponents(bin: Uint8ClampedArray, w: number, h: number): Box[] {
  const seen = new Uint8Array(bin.length);
  const boxes: Box[] = [];
  const qx: number[] = [];
  const qy: number[] = [];
  const push = (x: number, y: number) => {
    qx.push(x);
    qy.push(y);
  };
  const pop = () => ({ x: qx.pop()!, y: qy.pop()! });
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < h;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (bin[idx] === 1 && !seen[idx]) {
        let minx = x,
          maxx = x,
          miny = y,
          maxy = y;
        push(x, y);
        seen[idx] = 1;
        while (qx.length) {
          const { x: cx, y: cy } = pop();
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
            const nidx = ny * w + nx;
            if (bin[nidx] === 1 && !seen[nidx]) {
              seen[nidx] = 1;
              push(nx, ny);
            }
          }
        }
        const bw = maxx - minx + 1;
        const bh = maxy - miny + 1;
        if (bw >= 5 && bh >= 10) boxes.push({ x: minx, y: miny, w: bw, h: bh });
      }
    }
  }
  return boxes.sort((a, b) => (a.y === b.y ? a.x - b.x : a.y - b.y));
}

function cropTo28x28(src: ImageData, b: Box): Float32Array {
  const out = new Float32Array(28 * 28);
  const scale = Math.min(28 / b.w, 28 / b.h);
  const tw = Math.max(1, Math.floor(b.w * scale));
  const th = Math.max(1, Math.floor(b.h * scale));
  const ox = Math.floor((28 - tw) / 2);
  const oy = Math.floor((28 - th) / 2);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const sx = b.x + Math.floor(x / scale);
      const sy = b.y + Math.floor(y / scale);
      const idx = (sy * src.width + sx) * 4;
      const r = src.data[idx],
        g = src.data[idx + 1],
        bch = src.data[idx + 2];
      const gray = (r + g + bch) / 3 / 255;
      out[(oy + y) * 28 + (ox + x)] = gray;
    }
  }
  return out;
}

const OCRPanel: React.FC = () => {
  const [image, setImage] = useState<HTMLCanvasElement | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [text, setText] = useState<string>("");
  const uploadRef = useRef<HTMLInputElement | null>(null);

  const choose = (t: string) => {
    const c = drawTextToCanvas(t);
    setImage(c);
    const ctx = c.getContext("2d")!;
    const img = ctx.getImageData(0, 0, c.width, c.height);
    const bin = binarize(img);
    const comps = findComponents(bin, c.width, c.height);
    setBoxes(comps);
  };

  const run = async () => {
    if (!image) return;
    const ctx = image.getContext("2d")!;
    const img = ctx.getImageData(0, 0, image.width, image.height);
    let out = "";
    for (const b of boxes) {
      const crop = cropTo28x28(img, b);
      const c = getTrainerClient();
      c.predict(crop);
      // wait for prediction event
      const result = await new Promise<Float32Array>((resolve) => {
        const off = c.on("prediction", (p) => {
          resolve(p.probs);
          off();
        });
      });
      const arr = Array.from(result);
      let best = 0,
        bi = 0;
      for (let i = 0; i < arr.length; i++)
        if (arr[i] > best) {
          best = arr[i];
          bi = i;
        }
      const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
      out += chars[bi] ?? "?";
    }
    setText(out);
  };

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = Math.min(img.width, 1024);
      c.height = Math.floor((img.height / img.width) * c.width);
      const ctx = c.getContext("2d")!;
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      setImage(c);
      const id = ctx.getImageData(0, 0, c.width, c.height);
      const bin = binarize(id);
      const comps = findComponents(bin, c.width, c.height);
      setBoxes(comps);
    };
    img.src = URL.createObjectURL(f);
  };

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
        </div>
      </div>
    </section>
  );
};

export default OCRPanel;
