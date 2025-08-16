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

  const pageSize = 36;
  const [page, setPage] = useState(0);
  const totalPages = Math.max(1, Math.ceil(tiles.length / pageSize));
  const start = page * pageSize;
  const visible = tiles.slice(start, start + pageSize);

  return (
    <section style={{ padding: "1rem" }}>
      <h3>Anatomy</h3>
      {tiles.length === 0 ? (
        <p style={{ color: "var(--color-foreground-subtle)" }}>
          Start training to stream weight tiles.
        </p>
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(6, 1fr)",
              gap: 12,
            }}
          >
            {visible.map((t) => (
              <Heatmap key={t.name} grid={t.grid} title={t.name} />
            ))}
          </div>
          {totalPages > 1 ? (
            <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
              <button onClick={() => setPage((p) => Math.max(0, p - 1))}>
                Prev
              </button>
              <span style={{ color: "var(--color-foreground-subtle)" }}>
                Page {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              >
                Next
              </button>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
};

export default AnatomyPanel;
