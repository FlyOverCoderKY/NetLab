import React, { useEffect, useState } from "react";
import Heatmap from "./Heatmap";
import { getTrainerClient } from "../../worker/client";

type WeightTile = { name: string; grid: number[][] };

const AnatomyPanel: React.FC = () => {
  const [tiles, setTiles] = useState<WeightTile[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  useEffect(() => {
    const c = getTrainerClient();
    const offVis = c.on(
      "visuals",
      (payload: {
        weightsArr?: {
          name: string;
          width: number;
          height: number;
          data: Float32Array;
        }[];
        filtersArr?: {
          name: string;
          width: number;
          height: number;
          data: Float32Array;
        }[];
        activationsArr?: {
          layer: string;
          width: number;
          height: number;
          data: Float32Array;
        }[];
        overlaysArr?: {
          name: string;
          width: number;
          height: number;
          data: Float32Array;
        }[];
        // legacy nested arrays (fallback)
        weights?: { name: string; grid: number[][] }[];
        filters?: { name: string; grid: number[][] }[];
        activations?: { layer: string; grid: number[][] }[];
      }) => {
        const items: WeightTile[] = [];
        if (payload?.overlaysArr)
          items.push(
            ...payload.overlaysArr.map((t) => ({
              name: t.name,
              grid: toGrid(t.width, t.height, t.data),
            })),
          );
        // Prefer flat arrays when available
        const toGrid = (
          width: number,
          height: number,
          data: Float32Array,
        ): number[][] => {
          const g: number[][] = Array.from({ length: height }, () =>
            Array(width).fill(0),
          );
          for (let y = 0; y < height; y++)
            for (let x = 0; x < width; x++) g[y][x] = data[y * width + x];
          return g;
        };
        if (payload?.weightsArr)
          items.push(
            ...payload.weightsArr.map((t) => ({
              name: t.name,
              grid: toGrid(t.width, t.height, t.data),
            })),
          );
        if (payload?.filtersArr)
          items.push(
            ...payload.filtersArr.map((t) => ({
              name: t.name,
              grid: toGrid(t.width, t.height, t.data),
            })),
          );
        if (payload?.activationsArr)
          items.push(
            ...payload.activationsArr.map((t, i) => ({
              name: `${t.layer} ${i}`,
              grid: toGrid(t.width, t.height, t.data),
            })),
          );
        if (payload?.weights)
          items.push(
            ...payload.weights.map((w) => ({ name: w.name, grid: w.grid })),
          );
        if (payload?.filters)
          items.push(
            ...payload.filters.map((f) => ({ name: f.name, grid: f.grid })),
          );
        if (payload?.activations)
          items.push(
            ...payload.activations.map((a, i) => ({
              name: `${a.layer} ${i}`,
              grid: a.grid,
            })),
          );
        if (items.length) setTiles(items);
      },
    );
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
      <p style={{ color: "var(--color-foreground-subtle)", marginTop: 4 }}>
        Inspect learned patterns. Each tile visualizes a weight/filter; brighter
        pixels indicate stronger positive influence.
      </p>
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
            {visible.map((t, i) => {
              const globalIndex = start + i;
              const isSel = selectedIndex === globalIndex;
              return (
                <button
                  key={t.name}
                  onClick={() => setSelectedIndex(isSel ? null : globalIndex)}
                  aria-pressed={isSel}
                  style={{
                    padding: 0,
                    border: isSel
                      ? "2px solid var(--color-accent)"
                      : "1px solid var(--color-border)",
                    background: "transparent",
                  }}
                >
                  <Heatmap grid={t.grid} title={t.name} />
                </button>
              );
            })}
          </div>
          <div
            style={{
              marginTop: 8,
              display: "flex",
              gap: 8,
              alignItems: "center",
            }}
          >
            <button
              onClick={() => {
                if (selectedIndex == null) return;
                const classIndex = selectedIndex % 36;
                getTrainerClient().setWeights({
                  modelType: "softmax",
                  op: "zero-class",
                  classIndex,
                });
              }}
              disabled={selectedIndex == null}
            >
              Zero selected class
            </button>
            <button
              onClick={() => {
                if (selectedIndex == null) return;
                const classIndex = selectedIndex % 36;
                getTrainerClient().setWeights({
                  modelType: "softmax",
                  op: "randomize-class",
                  classIndex,
                });
              }}
              disabled={selectedIndex == null}
            >
              Randomize selected class
            </button>
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
