import { create } from "zustand";

export type AppMode = "softmax" | "mlp" | "cnn";
export type AppTab = "playground" | "train" | "anatomy" | "math";

export type DatasetParams = {
  seed: number;
  fontFamily: string;
  fontSize: number;
  thickness: number;
  jitterPx: number;
  rotationDeg: number;
  invert: boolean;
  noise: boolean;
};

export type TrainingParams = {
  learningRate: number;
  batchSize: number;
  optimizer: "sgd" | "adam";
  weightDecay?: number;
};

type State = {
  mode: AppMode;
  tab: AppTab;
  dataset: DatasetParams;
  training: TrainingParams;
};

type Actions = {
  setMode: (mode: AppMode) => void;
  setTab: (tab: AppTab) => void;
  setDataset: (update: Partial<DatasetParams>) => void;
  setTraining: (update: Partial<TrainingParams>) => void;
};

export const useAppStore = create<State & Actions>((set) => ({
  mode: "softmax",
  tab: "playground",
  dataset: {
    seed: 1234,
    fontFamily: "Inter",
    fontSize: 20,
    thickness: 20,
    jitterPx: 1,
    rotationDeg: 4,
    invert: false,
    noise: false,
  },
  training: {
    learningRate: 0.01,
    batchSize: 32,
    optimizer: "sgd",
    weightDecay: 0,
  },
  setMode: (mode) => set({ mode }),
  setTab: (tab) => set({ tab }),
  setDataset: (update) =>
    set((s) => ({ dataset: { ...s.dataset, ...update } })),
  setTraining: (update) =>
    set((s) => ({ training: { ...s.training, ...update } })),
}));
