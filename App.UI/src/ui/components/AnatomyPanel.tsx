import React, { useEffect, useState } from "react";
import Heatmap from "./Heatmap";
import { getTrainerClient } from "../../worker/client";

type WeightTile = { name: string; grid: number[][] };

const AnatomyPanel: React.FC = () => {
  const [tiles] = useState<WeightTile[]>([]);

  useEffect(() => {
    const c = getTrainerClient();
    const onVisuals = c.on("metrics", () => {
      // Request visuals periodically via a side-channel if implemented later
    });
    return () => onVisuals();
  }, []);

  // Placeholder: request visuals via direct call from worker model when available
  // For now, render nothing until we add a message to fetch visuals.

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Anatomy</h3>
      {tiles.length === 0 ? (
        <p style={{ color: "var(--color-foreground-subtle)" }}>
          Visuals will appear here as we wire up weight exports from the worker.
        </p>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, 1fr)",
            gap: 12,
          }}
        >
          {tiles.map((t) => (
            <Heatmap key={t.name} grid={t.grid} title={t.name} />
          ))}
        </div>
      )}
    </section>
  );
};

export default AnatomyPanel;
