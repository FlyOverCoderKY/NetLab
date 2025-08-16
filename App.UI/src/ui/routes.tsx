import React from "react";
import { useAppStore } from "../state/store";
import DatasetPanel from "./components/DatasetPanel";
import TrainPanel from "./components/TrainPanel";
import PlaygroundPanel from "./components/PlaygroundPanel";
import AnatomyPanel from "./components/AnatomyPanel";
import MathPanel from "./components/MathPanel";
import OCRPanel from "./components/OCRPanel";

export const RoutesView: React.FC = () => {
  const tab = useAppStore((s) => s.tab);
  const isActive = (t: string) => tab === (t as typeof tab);
  const panelStyle = (active: boolean) => ({
    display: active ? "block" : "none",
  });
  return (
    <div>
      <div
        style={panelStyle(isActive("playground"))}
        aria-hidden={!isActive("playground")}
      >
        <PlaygroundPanel />
      </div>
      <div
        style={panelStyle(isActive("dataset"))}
        aria-hidden={!isActive("dataset")}
      >
        <DatasetPanel />
      </div>
      <div
        style={panelStyle(isActive("train"))}
        aria-hidden={!isActive("train")}
      >
        <TrainPanel />
      </div>
      <div style={panelStyle(isActive("ocr"))} aria-hidden={!isActive("ocr")}>
        <OCRPanel />
      </div>
      <div
        style={panelStyle(isActive("anatomy"))}
        aria-hidden={!isActive("anatomy")}
      >
        <AnatomyPanel />
      </div>
      <div style={panelStyle(isActive("math"))} aria-hidden={!isActive("math")}>
        <MathPanel />
      </div>
    </div>
  );
};
