import React from "react";
import ModeSwitcher from "./ModeSwitcher";
import TrainerStatus from "./TrainerStatus";
import Tabs from "./Tabs";

const AppToolbar: React.FC = () => {
  return (
    <div
      role="navigation"
      aria-label="Application Controls"
      style={{
        background: "var(--color-panel-background)",
        borderBottom: "1px solid var(--color-border)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          padding: "8px 12px",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <ModeSwitcher />
        </div>
        <TrainerStatus />
      </div>
      <Tabs />
    </div>
  );
};

export default AppToolbar;
