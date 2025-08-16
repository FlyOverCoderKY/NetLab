export type TrainConfig = {
  modelType: "softmax" | "mlp" | "cnn";
  seed: number;
  batchSize: number;
  learningRate: number;
  optimizer: "sgd" | "adam";
  weightDecay?: number;
  steps: number;
  snapshotEvery: number;
  dataset?: {
    fontFamily: string;
    fontSize: number;
    thickness: number;
    jitterPx: number;
    rotationDeg: number;
    invert: boolean;
    noise: boolean;
    contentScale?: number;
    contentJitter?: number;
  };
};

export type SetWeightsPayload = {
  modelType: "softmax";
  op: "zero-class" | "randomize-class";
  classIndex: number; // 0..35
};

export type SetOverlayPayload = {
  enabled: boolean;
  classIndex?: number; // for softmax: 0..35
};

export type InMsg =
  | { type: "init"; payload: { backend: "webgl" | "wasm" } }
  | { type: "compile"; payload: TrainConfig }
  | { type: "step"; payload?: Record<string, never> }
  | { type: "run"; payload: { steps: number } }
  | { type: "pause" }
  | { type: "set-weights"; payload: SetWeightsPayload }
  | { type: "set-overlay"; payload: SetOverlayPayload }
  | { type: "get-weights" }
  | { type: "save-weights" }
  | { type: "load-weights"; payload: { state: unknown } }
  | { type: "switch-model"; payload: { modelType: TrainConfig["modelType"] } }
  | { type: "predict"; payload: { x: Float32Array } }
  | { type: "predict-batch"; payload: { x: Float32Array; n: number } }
  | { type: "dispose" };

export type OutMsg =
  | { type: "ready" }
  | { type: "compiled"; payload: { params: number } }
  | { type: "metrics"; payload: { step: number; loss: number; acc?: number } }
  | { type: "visuals"; payload: import("../models/types").Visuals }
  | { type: "confusion"; payload: { labels: string[]; matrix: number[][] } }
  | { type: "weights"; payload: { model: string; state: unknown } }
  | { type: "error"; payload: { message: string } }
  | { type: "prediction"; payload: { probs: Float32Array } }
  | { type: "prediction-batch"; payload: { probs: Float32Array; n: number } }
  | { type: "done"; payload?: Record<string, never> };
