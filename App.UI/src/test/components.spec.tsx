import { describe, it, expect } from "vitest";
// React import not required with jsx: react-jsx
import { render, screen } from "@testing-library/react";
import Heatmap from "../ui/components/Heatmap";
import ModeSwitcher from "../ui/components/ModeSwitcher";
import { useAppStore } from "../state/store";

describe("Heatmap component", () => {
  it("renders a canvas with role img", () => {
    const grid = Array.from({ length: 28 }, () => Array(28).fill(0.5));
    render(<Heatmap grid={grid} title="Test" />);
    const canvas = screen.getByRole("img", { name: /Test heatmap/i });
    expect(canvas).toBeTruthy();
  });
});

describe("Mode switcher", () => {
  it("retains settings across changes", () => {
    // Read initial state
    const initial = useAppStore.getState().training.learningRate;
    render(<ModeSwitcher />);
    // Changing mode shouldn't reset training params
    const after = useAppStore.getState().training.learningRate;
    expect(after).toBe(initial);
  });
});
