import React, { useEffect, useRef } from "react";

type HeatmapProps = {
  grid: number[][]; // values in [0,1]
  size?: number; // canvas render size in px (square)
  title?: string;
};

const Heatmap: React.FC<HeatmapProps> = ({ grid, size = 112, title }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const h = grid.length;
    const w = grid[0]?.length ?? 0;
    if (!w || !h) return;
    c.width = size;
    c.height = size;
    const ctx = c.getContext("2d");
    if (!ctx) return; // jsdom/no-canvas env
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = Math.max(0, Math.min(1, grid[y][x]));
        const i = (y * w + x) * 4;
        const g = Math.round(v * 255);
        img.data[i + 0] = g;
        img.data[i + 1] = g;
        img.data[i + 2] = g;
        img.data[i + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d")!;
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(off, 0, 0, w, h, 0, 0, size, size);
  }, [grid, size]);
  return (
    <figure style={{ margin: 0 }}>
      <canvas
        ref={ref}
        role="img"
        aria-label={title ? `${title} heatmap` : "model weights heatmap"}
        title={title ?? "Heatmap"}
      />
      {title && (
        <figcaption
          style={{ fontSize: 12, color: "var(--color-foreground-subtle)" }}
        >
          {title}
        </figcaption>
      )}
    </figure>
  );
};

export default Heatmap;
