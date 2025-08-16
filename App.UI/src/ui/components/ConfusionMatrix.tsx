import React, { useEffect, useState } from "react";
import { getTrainerClient } from "../../worker/client";

const cellSize = 12;

const ConfusionMatrix: React.FC = () => {
  const [labels, setLabels] = useState<string[]>([]);
  const [matrix, setMatrix] = useState<number[][]>([]);

  useEffect(() => {
    const c = getTrainerClient();
    const off = c.on("confusion", ({ labels, matrix }) => {
      setLabels(labels);
      setMatrix(matrix);
    });
    return () => off();
  }, []);

  if (!matrix.length) {
    return (
      <p style={{ color: "var(--color-foreground-subtle)" }}>
        Confusion matrix will appear during training.
      </p>
    );
  }

  const size = labels.length * cellSize;

  return (
    <div>
      <svg width={size} height={size} role="img" aria-label="confusion-matrix">
        {matrix.map((row, y) =>
          row.map((v, x) => {
            const c = Math.round(255 - Math.max(0, Math.min(1, v)) * 255);
            const fill = `rgb(${c},${c},${c})`;
            return (
              <rect
                key={`${x}-${y}`}
                x={x * cellSize}
                y={y * cellSize}
                width={cellSize}
                height={cellSize}
                fill={fill}
              />
            );
          }),
        )}
        <rect
          x={0.5}
          y={0.5}
          width={size - 1}
          height={size - 1}
          fill="none"
          stroke="var(--color-border)"
        />
      </svg>
    </div>
  );
};

export default ConfusionMatrix;
