import React from "react";
import { useAppStore } from "../state/store";
// import DatasetPanel from "./components/DatasetPanel";
import TrainPanel from "./components/TrainPanel";
import PlaygroundPanel from "./components/PlaygroundPanel";
import AnatomyPanel from "./components/AnatomyPanel";

const Placeholder: React.FC<{ title: string }> = ({ title }) => (
  <section style={{ padding: "1rem" }}>
    <h3>{title}</h3>
    <p>Coming soon.</p>
  </section>
);

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
      return <Placeholder title="Math" />;
    default:
      return null;
  }
};
