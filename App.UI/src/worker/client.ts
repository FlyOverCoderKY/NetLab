import type { InMsg, OutMsg, TrainConfig } from "./messages";
import type { Visuals } from "../models/types";

type ListenerMap = {
  ready: (() => void)[];
  error: ((m: string) => void)[];
  metrics: ((p: { step: number; loss: number; acc?: number }) => void)[];
  compiled: ((p: { params: number }) => void)[];
  done: (() => void)[];
  prediction: ((p: { probs: Float32Array }) => void)[];
  "prediction-batch": ((p: { probs: Float32Array; n: number }) => void)[];
  visuals: ((v: Visuals) => void)[];
  confusion: ((p: { labels: string[]; matrix: number[][] }) => void)[];
  weights: ((p: { model: string; state: unknown }) => void)[];
};

class TrainerClient {
  private worker: Worker;
  private listeners: ListenerMap = {
    ready: [],
    error: [],
    metrics: [],
    compiled: [],
    done: [],
    prediction: [],
    "prediction-batch": [],
    visuals: [],
    confusion: [],
    weights: [],
  };
  private isDisposed = false;

  constructor() {
    this.worker = new Worker(new URL("./trainer.ts", import.meta.url), {
      type: "module",
    });
    this.worker.addEventListener("message", (ev: MessageEvent<OutMsg>) => {
      const msg = ev.data;
      switch (msg.type) {
        case "ready":
          this.listeners.ready.forEach((cb) => cb());
          break;
        case "compiled":
          this.listeners.compiled.forEach((cb) => cb(msg.payload));
          break;
        case "metrics":
          this.listeners.metrics.forEach((cb) => cb(msg.payload));
          break;
        case "visuals":
          this.listeners.visuals.forEach((cb) => cb(msg.payload as Visuals));
          break;
        case "confusion":
          this.listeners.confusion.forEach((cb) =>
            cb(msg.payload as { labels: string[]; matrix: number[][] }),
          );
          break;
        case "prediction":
          this.listeners.prediction.forEach((cb) =>
            cb(msg.payload as { probs: Float32Array }),
          );
          break;
        case "prediction-batch":
          this.listeners["prediction-batch"].forEach((cb) =>
            cb(msg.payload as { probs: Float32Array; n: number }),
          );
          break;
        case "weights":
          this.listeners.weights.forEach((cb) =>
            cb(msg.payload as { model: string; state: unknown }),
          );
          break;
        case "done":
          this.listeners.done.forEach((cb) => cb());
          break;
        case "error":
          this.listeners.error.forEach((cb) => cb(msg.payload.message));
          break;
      }
    });
    this.init();
  }

  private init() {
    const msg: InMsg = { type: "init", payload: { backend: "wasm" } };
    this.worker.postMessage(msg);
  }

  on<T extends keyof ListenerMap>(
    type: T,
    cb: ListenerMap[T][number],
  ): () => void {
    const arr = this.listeners[type] as unknown as Array<
      ListenerMap[T][number]
    >;
    arr.push(cb as ListenerMap[T][number]);
    return () => {
      const idx = arr.indexOf(cb as ListenerMap[T][number]);
      if (idx >= 0) arr.splice(idx, 1);
    };
  }

  compile(cfg: TrainConfig) {
    const msg: InMsg = { type: "compile", payload: cfg };
    this.worker.postMessage(msg);
  }

  run(steps: number) {
    const msg: InMsg = { type: "run", payload: { steps } };
    this.worker.postMessage(msg);
  }

  step() {
    const msg: InMsg = { type: "step", payload: {} };
    this.worker.postMessage(msg);
  }

  pause() {
    const msg: InMsg = { type: "pause" } as InMsg;
    this.worker.postMessage(msg);
  }

  predict(x: Float32Array) {
    const msg: InMsg = { type: "predict", payload: { x } } as InMsg;
    // Use transfer list for performance
    this.worker.postMessage(msg, [x.buffer]);
  }

  predictBatch(x: Float32Array, n: number) {
    const msg: InMsg = {
      type: "predict-batch",
      payload: { x, n },
    } as InMsg;
    this.worker.postMessage(msg, [x.buffer]);
  }

  dispose() {
    if (this.isDisposed) return;
    this.isDisposed = true;
    const msg: InMsg = { type: "dispose" };
    this.worker.postMessage(msg);
    this.worker.terminate();
  }

  saveWeights() {
    const msg: InMsg = { type: "save-weights" } as InMsg;
    this.worker.postMessage(msg);
  }

  getWeights() {
    const msg: InMsg = { type: "get-weights" } as InMsg;
    this.worker.postMessage(msg);
  }

  loadWeights(state: unknown) {
    const msg: InMsg = { type: "load-weights", payload: { state } } as InMsg;
    this.worker.postMessage(msg);
  }

  switchModel(modelType: TrainConfig["modelType"]) {
    const msg: InMsg = {
      type: "switch-model",
      payload: { modelType },
    } as InMsg;
    this.worker.postMessage(msg);
  }

  setWeights(
    payload: InMsg & { type: "set-weights" } extends { payload: infer P }
      ? P
      : never,
  ) {
    const msg: InMsg = { type: "set-weights", payload } as InMsg;
    this.worker.postMessage(msg);
  }

  setOverlay(
    payload: InMsg & { type: "set-overlay" } extends { payload: infer P }
      ? P
      : never,
  ) {
    const msg: InMsg = { type: "set-overlay", payload } as InMsg;
    this.worker.postMessage(msg);
  }
}

let singleton: TrainerClient | null = null;
export function getTrainerClient(): TrainerClient {
  if (!singleton) singleton = new TrainerClient();
  return singleton;
}
