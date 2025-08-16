import React, { useEffect, useMemo } from "react";
import {
  CLASS_LIST,
  renderGlyphTo28x28,
  waitForFontsReady,
} from "../../data/generator";
import { XorShift32 } from "../../data/seed";
import { useAppStore } from "../../state/store";

const cellSize = 32;

const Slider: React.FC<{
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}> = ({ label, min, max, step, value, onChange }) => (
  <label style={{ display: "block", marginBottom: 8 }}>
    <span style={{ display: "inline-block", width: 160 }}>{label}</span>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ width: 240 }}
    />
    <span style={{ marginLeft: 8 }}>{value}</span>
  </label>
);

const DatasetPanel: React.FC = () => {
  const dataset = useAppStore((s) => s.dataset);
  const setDataset = useAppStore((s) => s.setDataset);

  useEffect(() => {
    waitForFontsReady().catch(() => void 0);
  }, []);

  const params = useMemo(
    () => ({
      fontFamily: dataset.fontFamily,
      fontSize: dataset.fontSize,
      thickness: dataset.thickness,
      jitterPx: dataset.jitterPx,
      rotationDeg: dataset.rotationDeg,
      invert: dataset.invert,
      noise: dataset.noise,
    }),
    [dataset],
  );

  const grid = useMemo(() => {
    const rng = new XorShift32(dataset.seed);
    return CLASS_LIST.slice(0, 36).map((ch) =>
      renderGlyphTo28x28(ch, params, rng),
    );
  }, [dataset.seed, params]);

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Dataset Preview</h3>
      <div
        style={{
          display: "flex",
          gap: 24,
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div>
          <Slider
            label="Seed"
            min={0}
            max={10000}
            step={1}
            value={dataset.seed}
            onChange={(v) => setDataset({ seed: v })}
          />
          <Slider
            label="Font Size"
            min={10}
            max={28}
            step={1}
            value={dataset.fontSize}
            onChange={(v) => setDataset({ fontSize: v })}
          />
          <Slider
            label="Thickness"
            min={8}
            max={28}
            step={1}
            value={dataset.thickness}
            onChange={(v) => setDataset({ thickness: v })}
          />
          <Slider
            label="Jitter (px)"
            min={0}
            max={3}
            step={1}
            value={dataset.jitterPx}
            onChange={(v) => setDataset({ jitterPx: v })}
          />
          <Slider
            label="Rotation (±deg)"
            min={0}
            max={15}
            step={1}
            value={dataset.rotationDeg}
            onChange={(v) => setDataset({ rotationDeg: v })}
          />
          <label style={{ display: "block", marginTop: 8 }}>
            <input
              type="checkbox"
              checked={dataset.invert}
              onChange={(e) => setDataset({ invert: e.target.checked })}
            />{" "}
            Invert
          </label>
          <label style={{ display: "block", marginTop: 8 }}>
            <input
              type="checkbox"
              checked={dataset.noise}
              onChange={(e) => setDataset({ noise: e.target.checked })}
            />{" "}
            Noise
          </label>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, 1fr)",
            gap: 8,
          }}
        >
          {grid.map((arr, i) => (
            <Canvas28 key={i} data={arr} label={CLASS_LIST[i]} />
          ))}
        </div>
      </div>
    </section>
  );
};

const Canvas28: React.FC<{ data: Float32Array; label: string }> = ({
  data,
  label,
}) => {
  const ref = React.useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = cellSize;
    c.height = cellSize;
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
    ctx.clearRect(0, 0, cellSize, cellSize);
    ctx.drawImage(off, 0, 0, 28, 28, 0, 0, cellSize, cellSize);
    ctx.fillStyle = "#666";
    ctx.font = "10px sans-serif";
    ctx.fillText(label, 2, cellSize - 2);
  }, [data, label]);
  return <canvas ref={ref} aria-label={`sample ${label}`} />;
};

export default DatasetPanel;
