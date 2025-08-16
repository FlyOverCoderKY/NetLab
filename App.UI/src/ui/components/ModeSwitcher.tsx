import React from "react";
import { useAppStore, AppMode } from "../../state/store";

const options: { value: AppMode; label: string }[] = [
  { value: "softmax", label: "Softmax" },
  { value: "mlp", label: "MLP" },
  { value: "cnn", label: "Tiny CNN" },
];

const ModeSwitcher: React.FC = () => {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  return (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      <span>Mode</span>
      <select
        aria-label="Model Mode"
        value={mode}
        onChange={(e) => setMode(e.target.value as AppMode)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
};

export default ModeSwitcher;
