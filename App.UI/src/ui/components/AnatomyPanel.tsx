import React, { useEffect, useState } from "react";
import Heatmap from "./Heatmap";
import { getTrainerClient } from "../../worker/client";

type WeightTile = { name: string; grid: number[][] };

const AnatomyPanel: React.FC = () => {
  const [tiles, setTiles] = useState<WeightTile[]>([]);

  useEffect(() => {
    const c = getTrainerClient();
    const offVis = c.on("visuals", (payload: { weights?: WeightTile[] }) => {
      if (payload?.weights) setTiles(payload.weights);
    });
    return () => {
      offVis();
    };
  }, []);

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Anatomy</h3>
      {tiles.length === 0 ? (
        <p style={{ color: "var(--color-foreground-subtle)" }}>
          Start training to stream weight tiles.
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
