import React from "react";
import { useAppStore } from "../state/store";
// import DatasetPanel from "./components/DatasetPanel";
import TrainPanel from "./components/TrainPanel";
import PlaygroundPanel from "./components/PlaygroundPanel";
import AnatomyPanel from "./components/AnatomyPanel";
import MathPanel from "./components/MathPanel";

export const RoutesView: React.FC = () => {
  const tab = useAppStore((s) => s.tab);
  switch (tab) {
    case "playground":
      return <PlaygroundPanel />;
    case "train":
      return <TrainPanel />;
    case "anatomy":
      return <AnatomyPanel />;
    case "math":
      return <MathPanel />;
    default:
      return null;
  }
};
