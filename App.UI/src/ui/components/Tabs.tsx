import React from "react";
import "./Tabs.css";
import { AppTab, useAppStore } from "../../state/store";

const tabs: { id: AppTab; label: string }[] = [
  { id: "playground", label: "Playground" },
  { id: "dataset", label: "Dataset" },
  { id: "train", label: "Train" },
  { id: "anatomy", label: "Anatomy" },
  { id: "math", label: "Math" },
];

const Tabs: React.FC = () => {
  const active = useAppStore((s) => s.tab);
  const setTab = useAppStore((s) => s.setTab);
  return (
    <nav aria-label="Main Tabs" className="tabs-nav">
      <ul className="tabs-list">
        {tabs.map((t) => (
          <li key={t.id}>
            <button
              onClick={() => setTab(t.id)}
              aria-current={active === t.id ? "page" : undefined}
              className="tab-btn"
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Tabs;
