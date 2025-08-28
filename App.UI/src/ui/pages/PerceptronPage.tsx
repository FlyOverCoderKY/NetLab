import React, { useMemo, useState } from "react";

const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));

const PerceptronPage: React.FC = () => {
  const [x1, setX1] = useState(0);
  const [x2, setX2] = useState(0);
  const [w1, setW1] = useState(1);
  const [w2, setW2] = useState(1);
  const [b, setB] = useState(0);
  const [activation, setActivation] = useState<"step" | "sigmoid">("step");

  const z = useMemo(() => w1 * x1 + w2 * x2 + b, [x1, x2, w1, w2, b]);
  const y = useMemo(
    () => (activation === "step" ? (z >= 0 ? 1 : 0) : 1 / (1 + Math.exp(-z))),
    [z, activation],
  );

  return (
    <div style={{ padding: 16, display: "grid", gap: 16 }}>
      <h2>Perceptron</h2>
      <div
        style={{
          display: "grid",
          gap: 12,
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        }}
      >
        <div>
          <label>Input x1: {x1.toFixed(2)}</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={x1}
            onChange={(e) => setX1(clamp(Number(e.target.value), -5, 5))}
          />
        </div>
        <div>
          <label>Input x2: {x2.toFixed(2)}</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={x2}
            onChange={(e) => setX2(clamp(Number(e.target.value), -5, 5))}
          />
        </div>
        <div>
          <label>Weight w1: {w1.toFixed(2)}</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={w1}
            onChange={(e) => setW1(clamp(Number(e.target.value), -5, 5))}
          />
        </div>
        <div>
          <label>Weight w2: {w2.toFixed(2)}</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={w2}
            onChange={(e) => setW2(clamp(Number(e.target.value), -5, 5))}
          />
        </div>
        <div>
          <label>Bias b: {b.toFixed(2)}</label>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={b}
            onChange={(e) => setB(clamp(Number(e.target.value), -5, 5))}
          />
        </div>
        <div>
          <label>Activation</label>
          <select
            value={activation}
            onChange={(e: React.ChangeEvent<HTMLSelectElement>) =>
              setActivation(e.target.value as "step" | "sigmoid")
            }
          >
            <option value="step">Step</option>
            <option value="sigmoid">Sigmoid</option>
          </select>
        </div>
      </div>
      <div>
        <strong>z = w1*x1 + w2*x2 + b:</strong> {z.toFixed(3)}
      </div>
      <div>
        <strong>Output y:</strong> {y.toFixed(3)}
      </div>
    </div>
  );
};

export default PerceptronPage;
