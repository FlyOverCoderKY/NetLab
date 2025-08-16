import React, { useCallback, useEffect, useMemo, useState } from "react";
import { CLASS_LIST, renderGlyphTo28x28 } from "../../data/generator";
import { XorShift32 } from "../../data/seed";
import { useAppStore } from "../../state/store";
import { getTrainerClient } from "../../worker/client";

const Canvas28: React.FC<{ data: Float32Array }> = ({ data }) => {
  const [ref, setRef] = useState<HTMLCanvasElement | null>(null);
  useEffect(() => {
    if (!ref) return;
    ref.width = 140;
    ref.height = 140;
    const ctx = ref.getContext("2d")!;
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
    ctx.clearRect(0, 0, 140, 140);
    ctx.drawImage(off, 0, 0, 28, 28, 0, 0, 140, 140);
  }, [data, ref]);
  return <canvas ref={setRef} aria-label="playground-sample" />;
};

const PlaygroundPanel: React.FC = () => {
  const dataset = useAppStore((s) => s.dataset);
  const [idx, setIdx] = useState(0);
  const [probs, setProbs] = useState<Float32Array | null>(null);
  const [auto, setAuto] = useState(false);
  const [intervalSec, setIntervalSec] = useState(2);

  const sample = useMemo(() => {
    const rng = new XorShift32(dataset.seed + idx);
    const ch = CLASS_LIST[idx % CLASS_LIST.length];
    return renderGlyphTo28x28(ch, dataset, rng);
  }, [dataset, idx]);

  useEffect(() => {
    const c = getTrainerClient();
    const off = c.on("prediction", (payload) => setProbs(payload.probs));
    return () => off();
  }, []);

  const predict = useCallback(() => {
    const c = getTrainerClient();
    c.predict(sample);
  }, [sample]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(
      () => {
        setIdx((i) => (i + 1) % CLASS_LIST.length);
        predict();
      },
      Math.max(500, intervalSec * 1000),
    );
    return () => clearInterval(id);
  }, [auto, intervalSec, predict]);

  const topk = useMemo(() => {
    if (!probs) return [] as Array<{ label: string; p: number }>;
    const pairs = Array.from(probs).map((p, i) => ({ i, p }));
    pairs.sort((a, b) => b.p - a.p);
    return pairs.slice(0, 5).map(({ i, p }) => ({ label: CLASS_LIST[i], p }));
  }, [probs]);

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Playground</h3>
      <div
        style={{
          display: "flex",
          gap: 16,
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div>
          <Canvas28 data={sample} />
        </div>
        <div>
          <h4>Top-5</h4>
          {topk.length === 0 ? (
            <p style={{ color: "var(--color-foreground-subtle)" }}>
              No prediction yet.
            </p>
          ) : (
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {topk.map((t) => (
                <li
                  key={t.label}
                  style={{ display: "flex", gap: 8, alignItems: "center" }}
                >
                  <span style={{ width: 16, textAlign: "right" }}>
                    {t.label}
                  </span>
                  <div
                    style={{
                      width: 160,
                      height: 8,
                      background: "var(--color-border)",
                    }}
                  >
                    <div
                      style={{
                        width: `${Math.round(t.p * 100)}%`,
                        height: 8,
                        background: "var(--color-accent)",
                      }}
                    />
                  </div>
                  <span style={{ width: 48 }}>{(t.p * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div
        style={{
          marginTop: 12,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <button onClick={() => setIdx((i) => (i + 1) % CLASS_LIST.length)}>
          Next
        </button>
        <button onClick={predict}>Predict</button>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <input
            type="checkbox"
            checked={auto}
            onChange={(e) => setAuto(e.target.checked)}
          />
          <span>Auto-cycle</span>
        </label>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>Every</span>
          <input
            type="number"
            min={1}
            max={10}
            value={intervalSec}
            onChange={(e) =>
              setIntervalSec(
                Math.max(1, Math.min(10, Number(e.target.value) || 1)),
              )
            }
            style={{ width: 48 }}
            aria-label="Auto-cycle interval seconds"
          />
          <span>s</span>
        </label>
      </div>
    </section>
  );
};

export default PlaygroundPanel;
