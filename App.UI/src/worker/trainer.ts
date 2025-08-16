/// <reference lib="webworker" />
// NOTE: TF.js imports are deferred until needed in later phases to avoid type resolution during scaffolding.
import type { InMsg, OutMsg, TrainConfig } from "./messages";

let isRunning = false;
let disposed = false;
let stepCounter = 0;
let lastLoss = 1;
let snapshotEvery = 10;
let rafId: number | null = null;
let compiled = false;

function handleInit(_: { backend: "webgl" | "wasm" }) {
  void _;
  postMessage({ type: "ready" } as OutMsg);
}

function handleCompile(cfg: TrainConfig) {
  snapshotEvery = Math.max(1, (cfg.snapshotEvery as number) | 0);
  stepCounter = 0;
  lastLoss = 1;
  compiled = true;
  postMessage({
    type: "compiled",
    payload: { params: 36 * 784 + 36 },
  } as OutMsg);
}

function postMetrics() {
  postMessage({
    type: "metrics",
    payload: { step: stepCounter, loss: lastLoss },
  } as OutMsg);
}

function handleStep() {
  if (!compiled) return;
  stepCounter += 1;
  lastLoss = Math.max(0, lastLoss * 0.99 + Math.random() * 0.005);
  if (stepCounter % snapshotEvery === 0) {
    postMetrics();
  }
}

function handleRun(totalSteps: number) {
  if (!compiled) return;
  let remaining = totalSteps;
  if (rafId != null) cancelAnimationFrame(rafId);
  isRunning = true;
  const tick = () => {
    if (!isRunning) return;
    const batch = Math.min(remaining, 10);
    for (let i = 0; i < batch; i++) handleStep();
    remaining -= batch;
    if (remaining <= 0) {
      isRunning = false;
      postMessage({ type: "done", payload: {} } as OutMsg);
      return;
    }
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}

self.onmessage = (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  if (disposed) return;
  switch (msg.type) {
    case "init": {
      handleInit(msg.payload);
      break;
    }
    case "compile": {
      handleCompile(msg.payload as TrainConfig);
      break;
    }
    case "run": {
      handleRun(msg.payload.steps);
      break;
    }
    case "step": {
      handleStep();
      break;
    }
    case "pause": {
      isRunning = false;
      if (rafId != null) cancelAnimationFrame(rafId);
      rafId = null;
      break;
    }
    case "dispose": {
      disposed = true;
      isRunning = false;
      if (rafId != null) cancelAnimationFrame(rafId);
      rafId = null;
      // Best effort cleanup
      close();
      break;
    }
    default: {
      postMessage({
        type: "error",
        payload: { message: `Unhandled message: ${msg.type}` },
      } as OutMsg);
    }
  }
};

export {};
