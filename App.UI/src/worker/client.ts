import type { InMsg, OutMsg, TrainConfig } from "./messages";

type ListenerMap = {
  ready: (() => void)[];
  error: ((m: string) => void)[];
  metrics: ((p: { step: number; loss: number; acc?: number }) => void)[];
  compiled: ((p: { params: number }) => void)[];
  done: (() => void)[];
};

class TrainerClient {
  private worker: Worker;
  private listeners: ListenerMap = {
    ready: [],
    error: [],
    metrics: [],
    compiled: [],
    done: [],
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

  dispose() {
    if (this.isDisposed) return;
    this.isDisposed = true;
    const msg: InMsg = { type: "dispose" };
    this.worker.postMessage(msg);
    this.worker.terminate();
  }
}

let singleton: TrainerClient | null = null;
export function getTrainerClient(): TrainerClient {
  if (!singleton) singleton = new TrainerClient();
  return singleton;
}
