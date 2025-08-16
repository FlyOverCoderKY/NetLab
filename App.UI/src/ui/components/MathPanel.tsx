import React from "react";
import { useAppStore } from "../../state/store";

const Eq: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <code
    style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
  >
    {children}
  </code>
);

const MathPanel: React.FC = () => {
  const mode = useAppStore((s) => s.mode);
  return (
    <section style={{ padding: "1rem", lineHeight: 1.6 }}>
      <h3>Math</h3>
      {mode === "softmax" && (
        <div>
          <p>
            Softmax regression: <Eq>z = W x + b</Eq>, <Eq>p = softmax(z)</Eq>
          </p>
          <p>
            Cross-entropy loss: <Eq>L = -log p_y</Eq>
          </p>
          <p>
            Gradients: <Eq>dL/dW = (p - y) x^T</Eq>, <Eq>dL/db = p - y</Eq>
          </p>
        </div>
      )}
      {mode === "mlp" && (
        <div>
          <p>
            MLP: <Eq>h = ReLU(W1 x + b1)</Eq>, <Eq>z = W2 h + b2</Eq>,{" "}
            <Eq>p = softmax(z)</Eq>
          </p>
          <p>Backprop follows dense layer chain rule.</p>
        </div>
      )}
      {mode === "cnn" && (
        <div>
          <p>
            Tiny CNN: <Eq>h = ReLU(x * K + b)</Eq> → MaxPool(2) → Flatten →
            Dense → Softmax
          </p>
          <p>Discuss locality and parameter sharing.</p>
        </div>
      )}
      <p style={{ color: "var(--color-foreground-subtle)", marginTop: 12 }}>
        Tip: Use Step to correlate updates with the equations.
      </p>
    </section>
  );
};

export default MathPanel;
