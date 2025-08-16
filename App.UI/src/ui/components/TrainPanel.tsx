import React, { useEffect, useRef, useState } from "react";
import { useAppStore } from "../../state/store";
import { getTrainerClient } from "../../worker/client";

type Metric = { step: number; loss: number };

const TrainPanel: React.FC = () => {
  const training = useAppStore((s) => s.training);
  const [running, setRunning] = useState(false);
  const [metrics, setMetrics] = useState<Metric[]>([]);

  const start = () => {
    if (running) return;
    const client = getTrainerClient();
    client.compile({
      modelType: "softmax",
      seed: 1234,
      batchSize: training.batchSize,
      learningRate: training.learningRate,
      optimizer: training.optimizer,
      weightDecay: training.weightDecay,
      steps: 0,
      snapshotEvery: 10,
    });
    client.run(5000);
    setRunning(true);
  };
  const pause = () => {
    getTrainerClient().pause();
    setRunning(false);
  };
  const stepOnce = () => {
    const client = getTrainerClient();
    client.compile({
      modelType: "softmax",
      seed: 1234,
      batchSize: training.batchSize,
      learningRate: training.learningRate,
      optimizer: training.optimizer,
      weightDecay: training.weightDecay,
      steps: 0,
      snapshotEvery: 1,
    });
    client.step();
  };

  useEffect(() => {
    const client = getTrainerClient();
    const offM = client.on("metrics", ({ step, loss }) => {
      setMetrics((m) => [...m, { step, loss }].slice(-500));
    });
    const offD = client.on("done", () => setRunning(false));
    return () => {
      offM();
      offD();
    };
  }, []);

  const last = metrics.length ? metrics[metrics.length - 1] : undefined;

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Train</h3>
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          margin: "8px 0",
        }}
      >
        <button onClick={start} disabled={running}>
          Start
        </button>
        <button onClick={pause} disabled={!running}>
          Pause
        </button>
        <button onClick={stepOnce} disabled={running}>
          Step
        </button>
        <span style={{ color: "var(--color-foreground-subtle)" }}>
          LR {training.learningRate} · Batch {training.batchSize} · Optimizer{" "}
          {training.optimizer}
        </span>
      </div>
      <div aria-live="polite">
        {last ? `Step ${last.step} — Loss ${last.loss.toFixed(4)}` : "Idle"}
      </div>
      <MiniChart data={metrics} />
    </section>
  );
};

const MiniChart: React.FC<{ data: Metric[] }> = ({ data }) => {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ctx = el.getContext("2d")!;
    const w = el.width;
    const h = el.height;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;
    const min = Math.min(...data.map((d) => d.loss));
    const max = Math.max(...data.map((d) => d.loss));
    const xs = (i: number) => (i / (data.length - 1)) * (w - 20) + 10;
    const ys = (v: number) =>
      h - 10 - ((v - min) / Math.max(1e-6, max - min)) * (h - 20);
    ctx.strokeStyle = "#60a5fa";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = xs(i);
      const y = ys(d.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.strokeStyle = "var(--color-border)";
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }, [data]);
  return (
    <canvas ref={ref} width={500} height={160} style={{ maxWidth: "100%" }} />
  );
};

export default TrainPanel;
