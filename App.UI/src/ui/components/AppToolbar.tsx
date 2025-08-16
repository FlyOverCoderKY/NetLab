import React from "react";
import ModeSwitcher from "./ModeSwitcher";
import TrainerStatus from "./TrainerStatus";
import Tabs from "./Tabs";
import { useTheme } from "../../context/ThemeContext";

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
          <HighContrastToggle />
        </div>
        <TrainerStatus />
      </div>
      <Tabs />
    </div>
  );
};

export default AppToolbar;

const HighContrastToggle: React.FC = () => {
  const theme = useTheme();
  return (
    <label
      style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
      title="High contrast mode"
    >
      <input
        type="checkbox"
        checked={theme.highContrast}
        onChange={(e) => theme.setHighContrast(e.target.checked)}
        aria-label="Toggle high contrast mode"
      />
      <span style={{ color: "var(--color-foreground-subtle)", fontSize: 12 }}>
        High contrast
      </span>
    </label>
  );
};
