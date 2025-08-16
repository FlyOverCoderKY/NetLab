export type TrainConfig = {
  modelType: "softmax" | "mlp" | "cnn";
  seed: number;
  batchSize: number;
  learningRate: number;
  optimizer: "sgd" | "adam";
  weightDecay?: number;
  steps: number;
  snapshotEvery: number;
};

export type InMsg =
  | { type: "init"; payload: { backend: "webgl" | "wasm" } }
  | { type: "compile"; payload: TrainConfig }
  | { type: "step"; payload?: Record<string, never> }
  | { type: "run"; payload: { steps: number } }
  | { type: "pause" }
  | { type: "set-weights"; payload: unknown }
  | { type: "switch-model"; payload: { modelType: TrainConfig["modelType"] } }
  | { type: "predict"; payload: { x: Float32Array } }
  | { type: "dispose" };

export type OutMsg =
  | { type: "ready" }
  | { type: "compiled"; payload: { params: number } }
  | { type: "metrics"; payload: { step: number; loss: number; acc?: number } }
  | { type: "visuals"; payload: import("../models/types").Visuals }
  | { type: "confusion"; payload: { labels: string[]; matrix: number[][] } }
  | { type: "weights"; payload: unknown }
  | { type: "error"; payload: { message: string } }
  | { type: "prediction"; payload: { probs: Float32Array } }
  | { type: "done"; payload?: Record<string, never> };
