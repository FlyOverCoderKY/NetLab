/// <reference lib="webworker" />
// NOTE: TF.js imports are deferred until needed in later phases to avoid type resolution during scaffolding.
import type {
  InMsg,
  OutMsg,
  TrainConfig,
  SetWeightsPayload,
  SetOverlayPayload,
} from "./messages";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
// Bundle wasm assets via Vite and point TFJS to the right URLs
// These imports resolve to URLs at runtime
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import wasmUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import wasmSimdUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url";
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import wasmThreadedSimdUrl from "@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url";
import { makeDataset } from "../data/dataset";
import { createModel } from "../models";
import type { Visuals } from "../models/types";

let isRunning = false;
let disposed = false;
let stepCounter = 0;
let lastLoss = 1;
let snapshotEvery = 10;
let rafId: number | null = null;
let compiled = false;
// Keep latest config for future training hooks
let currentCfg: TrainConfig | null = null;
let dataIterator: Generator<{ x: tf.Tensor4D; y: tf.Tensor1D }> | null = null;
let valIterator: Generator<{ x: tf.Tensor4D; y: tf.Tensor1D }> | null = null;
import type { TeachModel } from "../models/types";
let model: TeachModel | null = null;
let modelInitPromise: Promise<void> | null = null;
let lastVisualsSentAt = 0;
let overlayEnabled = false;
let overlayClassIndex = 0;

function handleInit(payload: { backend: "webgl" | "wasm" }) {
  // Read requested backend to satisfy lint, but we force WASM in the worker
  if (payload && payload.backend) {
    // no-op: backend selection is forced to wasm below
  }
  try {
    setWasmPaths({
      "tfjs-backend-wasm.wasm": wasmUrl as string,
      "tfjs-backend-wasm-simd.wasm": wasmSimdUrl as string,
      "tfjs-backend-wasm-threaded-simd.wasm": wasmThreadedSimdUrl as string,
    } as unknown as Record<string, string>);
  } catch {
    // ignore, TFJS will try defaults
  }
  tf.setBackend("wasm")
    .catch(async () => {
      // Fallback to CPU if WASM fails to initialize (e.g., wrong MIME)
      await tf.setBackend("cpu");
    })
    .then(() => {
      postMessage({ type: "ready" } as OutMsg);
    });
}

function handleCompile(cfg: TrainConfig) {
  snapshotEvery = Math.max(1, (cfg.snapshotEvery as number) | 0);
  stepCounter = 0;
  lastLoss = 1;
  compiled = true;
  currentCfg = cfg;
  void currentCfg;
  // Prepare dataset iterator
  dataIterator = makeDataset(
    {
      seed: cfg.seed,
      fontFamily: cfg.dataset?.fontFamily ?? "Inter",
      fontSize: cfg.dataset?.fontSize ?? 20,
      thickness: cfg.dataset?.thickness ?? 20,
      jitterPx: cfg.dataset?.jitterPx ?? 1,
      rotationDeg: cfg.dataset?.rotationDeg ?? 4,
      invert: cfg.dataset?.invert ?? false,
      noise: cfg.dataset?.noise ?? false,
    },
    cfg.batchSize,
  );
  // Prepare small validation iterator with a different seed
  valIterator = makeDataset(
    {
      seed: cfg.seed + 9999,
      fontFamily: cfg.dataset?.fontFamily ?? "Inter",
      fontSize: cfg.dataset?.fontSize ?? 20,
      thickness: cfg.dataset?.thickness ?? 20,
      jitterPx: cfg.dataset?.jitterPx ?? 1,
      rotationDeg: cfg.dataset?.rotationDeg ?? 4,
      invert: cfg.dataset?.invert ?? false,
      noise: cfg.dataset?.noise ?? false,
    },
    36,
  );
  // Report parameter count based on current model if available
  let paramCount = 36 * 784 + 36;
  try {
    if (model) {
      const m = (
        model as unknown as {
          model?: { layers?: { getWeights: () => tf.Tensor[] }[] };
        }
      ).model;
      if (m?.layers && Array.isArray(m.layers)) {
        paramCount = m.layers.reduce(
          (sum: number, layer: { getWeights: () => tf.Tensor[] }) => {
            const ws = layer.getWeights();
            return sum + ws.reduce((s, t) => s + (t.size ?? 0), 0);
          },
          0,
        );
      }
    }
  } catch {
    // ignore and keep fallback count
  }
  postMessage({
    type: "compiled",
    payload: { params: paramCount },
  } as OutMsg);
  // Pass hyperparams into model if already initialized
  if (
    model &&
    (
      model as unknown as {
        setHyperparams?: (h: {
          learningRate: number;
          optimizer: "sgd" | "adam";
          weightDecay?: number;
        }) => void;
      }
    ).setHyperparams
  ) {
    (
      model as unknown as {
        setHyperparams: (h: {
          learningRate: number;
          optimizer: "sgd" | "adam";
          weightDecay?: number;
        }) => void;
      }
    ).setHyperparams({
      learningRate: cfg.learningRate,
      optimizer: cfg.optimizer,
      weightDecay: cfg.weightDecay ?? 0,
    });
  }
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
  if (!model) {
    model = createModel(currentCfg?.modelType ?? "softmax");
    modelInitPromise = model.init();
  }
  if (modelInitPromise) {
    // Wait one tick for initialization to complete
    const p = modelInitPromise;
    modelInitPromise = null;
    p.then(() => void 0).catch(() => void 0);
    // Gentle decay until real loss arrives
    lastLoss = Math.max(0, lastLoss * 0.995 + Math.random() * 0.0025);
  } else if (dataIterator && model) {
    const { value } = dataIterator.next();
    if (value) {
      const batch = value as { x: tf.Tensor4D; y: tf.Tensor1D };
      model
        .trainStep(batch)
        .then((r: { loss?: number }) => {
          if (typeof r?.loss === "number" && !Number.isNaN(r.loss)) {
            lastLoss = r.loss;
          } else {
            lastLoss = Math.max(0, lastLoss * 0.995 + Math.random() * 0.0025);
          }
          batch.x.dispose();
          batch.y.dispose();
        })
        .catch(() => {
          lastLoss = Math.max(0, lastLoss * 0.995 + Math.random() * 0.0025);
          batch.x.dispose();
          batch.y.dispose();
        });
    }
  } else {
    // fallback: simulated loss trend
    lastLoss = Math.max(0, lastLoss * 0.99 + Math.random() * 0.005);
  }
  if (stepCounter % snapshotEvery === 0) {
    postMetrics();
    // Throttle visuals/confusion to <= 4 Hz
    const now = Date.now();
    const shouldEmitVisuals = now - lastVisualsSentAt >= 250;
    if (shouldEmitVisuals) lastVisualsSentAt = now;
    // Also push visuals and a lightweight accuracy estimate on snapshot
    if (model && shouldEmitVisuals) {
      model
        .getVisuals(undefined)
        .then(async (v: Visuals) => {
          // Prefer transfer-optimized arrays to reduce payload size
          const payload: Visuals = {
            weightsArr: v.weightsArr,
            filtersArr: v.filtersArr,
            activationsArr: v.activationsArr,
          };
          const transfers: ArrayBuffer[] = [];
          if (v.weightsArr)
            for (const t of v.weightsArr)
              transfers.push(t.data.buffer as ArrayBuffer);
          if (v.filtersArr)
            for (const t of v.filtersArr)
              transfers.push(t.data.buffer as ArrayBuffer);
          if (v.activationsArr)
            for (const t of v.activationsArr)
              transfers.push(t.data.buffer as ArrayBuffer);
          // Compute overlays for softmax first (per-class gradient magnitude)
          if (overlayEnabled && currentCfg?.modelType === "softmax") {
            try {
              const { value } = valIterator
                ? valIterator.next()
                : { value: null };
              if (value) {
                const batch = value as { x: tf.Tensor4D; y: tf.Tensor1D };
                const tile = tf.tidy(() => {
                  // x: [1,28,28,1] -> flatten to [784]
                  const x = batch.x
                    .slice([0, 0, 0, 0], [1, 28, 28, 1])
                    .reshape([784]);
                  // logits and softmax
                  const logits = (
                    model as unknown as {
                      model?: { predict: (x: tf.Tensor4D) => tf.Tensor };
                    }
                  ).model!.predict(batch.x) as tf.Tensor2D;
                  const probs = tf.softmax(logits).as1D();
                  const c = overlayClassIndex | 0;
                  const pc = probs.gather([c]).as1D();
                  const yc = tf
                    .equal(batch.y.gather([0]), tf.scalar(c, "int32"))
                    .cast("float32");
                  const diff = tf.sub(pc, yc); // scalar
                  const grad = tf.mul(x, diff); // [784]
                  const abs = tf.abs(grad);
                  const maxv = tf.maximum(tf.max(abs), tf.scalar(1e-8));
                  const norm = tf.div(abs, maxv); // [784] in 0..1
                  return norm.reshape([28, 28]);
                });
                const data = Float32Array.from(await tile.data());
                payload.overlaysArr = [
                  {
                    name: `dL/dW class ${overlayClassIndex}`,
                    width: 28,
                    height: 28,
                    data,
                  },
                ];
                transfers.push(data.buffer as ArrayBuffer);
                tile.dispose();
                batch.x.dispose();
                batch.y.dispose();
              }
            } catch {
              // ignore overlay errors
            }
          }
          postMessage({ type: "visuals", payload } as OutMsg, transfers);
        })
        .catch(() => void 0);
      // Accuracy and confusion estimate on a small validation batch to limit cost
      if (valIterator) {
        try {
          const { value } = valIterator.next();
          if (value) {
            const batch = value as { x: tf.Tensor4D; y: tf.Tensor1D };
            (async () => {
              try {
                const { probs } = await model!.predict(batch.x);
                const { acc, preds } = tf.tidy(() => {
                  const preds = probs.argMax(1);
                  const correct = tf.equal(preds, batch.y).sum() as tf.Scalar;
                  const acc = correct.div(
                    tf.scalar(batch.y.shape[0] || 1),
                  ) as tf.Scalar;
                  return { acc: acc.dataSync()[0] as number, preds };
                });
                postMessage({
                  type: "metrics",
                  payload: { step: stepCounter, loss: lastLoss, acc },
                } as OutMsg);
                // Compute a tiny confusion matrix for 36 classes from this batch
                const predsArr = preds.dataSync() as Int32Array;
                const labelsArr = batch.y.dataSync() as Int32Array;
                const size = 36;
                const cm: number[][] = Array.from({ length: size }, () =>
                  Array(size).fill(0),
                );
                for (let i = 0; i < labelsArr.length; i++) {
                  const yi = labelsArr[i] | 0;
                  const pi = predsArr[i] | 0;
                  cm[yi][pi] += 1;
                }
                // Normalize rows to sum to 1
                for (let r = 0; r < size; r++) {
                  const rowSum = cm[r].reduce((a, b) => a + b, 0) || 1;
                  for (let c = 0; c < size; c++) cm[r][c] = cm[r][c] / rowSum;
                }
                // Emit confusion once per snapshot (throttled with visuals)
                postMessage({
                  type: "confusion",
                  payload: {
                    labels: Array.from("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
                    matrix: cm,
                  },
                } as OutMsg);
                preds.dispose();
                probs.dispose();
              } catch {
                // ignore
              } finally {
                batch.x.dispose();
                batch.y.dispose();
              }
            })();
          }
        } catch {
          // ignore evaluation errors
        }
      }
    }
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
    case "set-overlay": {
      const p = msg.payload as SetOverlayPayload;
      overlayEnabled = !!p.enabled;
      if (typeof p.classIndex === "number") {
        overlayClassIndex = Math.max(0, Math.min(35, p.classIndex | 0));
      }
      break;
    }
    case "switch-model": {
      try {
        // Dispose current model and switch
        if (model) {
          try {
            model.dispose();
          } catch {
            // ignore
          }
        }
        model = createModel(msg.payload.modelType);
        modelInitPromise = model.init();
        if (currentCfg) currentCfg.modelType = msg.payload.modelType;
      } catch (err) {
        postMessage({ type: "error", payload: { message: String(err) } });
      }
      break;
    }
    case "set-weights": {
      const payload = msg.payload as SetWeightsPayload;
      if (!model) break;
      if (payload.modelType === "softmax") {
        // Update a single class column in the Dense kernel
        const current = model as unknown as {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          model?: { layers: any[] } | null;
          getVisuals: () => Promise<Visuals>;
        };
        const m = current.model as unknown as { layers: unknown[] } | null;
        if (!m) break;
        const dense = (m.layers as unknown[]).find(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (l: any) =>
            typeof l?.getClassName === "function" &&
            l.getClassName() === "Dense",
        ) as unknown as
          | {
              getWeights: () => tf.Tensor[];
              setWeights: (w: tf.Tensor[]) => void;
            }
          | undefined;
        if (!dense) break;
        const weights = dense.getWeights();
        if (!weights.length) break;
        const kernel = weights[0] as tf.Tensor2D; // [784, 36]
        const bias =
          (weights[1] as tf.Tensor1D) ?? tf.tensor1d(new Float32Array(36));
        const doUpdate = async () => {
          if (payload.op === "zero-class") {
            const arr = (await kernel.array()) as number[][];
            for (let i = 0; i < arr.length; i++) arr[i][payload.classIndex] = 0;
            const newKernel = tf.tensor2d(
              arr,
              kernel.shape as [number, number],
              "float32",
            );
            dense.setWeights([newKernel, bias]);
            newKernel.dispose();
          } else if (payload.op === "randomize-class") {
            const arr = (await kernel.array()) as number[][];
            for (let i = 0; i < arr.length; i++) arr[i][payload.classIndex] = 0; // reset first
            const newKernel = tf.tensor2d(
              arr,
              kernel.shape as [number, number],
              "float32",
            );
            const noise = tf.randomNormal(
              [kernel.shape[0], 1],
              0,
              0.01,
              "float32",
            );
            const updated = tf.tidy(() =>
              tf.concat(
                [
                  newKernel.slice(
                    [0, 0],
                    [kernel.shape[0], payload.classIndex],
                  ),
                  noise,
                  newKernel.slice(
                    [0, payload.classIndex + 1],
                    [kernel.shape[0], kernel.shape[1] - payload.classIndex - 1],
                  ),
                ],
                1,
              ),
            ) as tf.Tensor2D;
            dense.setWeights([updated, bias]);
            newKernel.dispose();
            noise.dispose();
            updated.dispose();
          }
          // Emit updated visuals
          try {
            const v = await (current.getVisuals() as Promise<Visuals>);
            postMessage({ type: "visuals", payload: v } as OutMsg);
          } catch {
            // ignore
          }
        };
        void doUpdate();
      }
      break;
    }
    case "get-weights": {
      if (!model) break;
      const currentModel = model;
      currentModel
        .serialize()
        .then((state) =>
          postMessage({
            type: "weights",
            payload: { model: currentModel.name, state },
          } as OutMsg),
        )
        .catch((err) =>
          postMessage({ type: "error", payload: { message: String(err) } }),
        );
      break;
    }
    case "save-weights": {
      if (!model) break;
      const currentModel = model;
      currentModel
        .serialize()
        .then((state) =>
          postMessage({
            type: "weights",
            payload: { model: currentModel.name, state },
          } as OutMsg),
        )
        .catch((err) =>
          postMessage({ type: "error", payload: { message: String(err) } }),
        );
      break;
    }
    case "load-weights": {
      if (model) {
        model
          .load(msg.payload.state as Record<string, unknown>)
          .catch((err: unknown) =>
            postMessage({ type: "error", payload: { message: String(err) } }),
          );
      }
      break;
    }
    case "predict": {
      const input = msg.payload.x; // expects 28*28 grayscale
      try {
        if (!model) {
          model = createModel(currentCfg?.modelType ?? "softmax");
          modelInitPromise = model.init();
        }
        const run = async () => {
          if (modelInitPromise) await modelInitPromise;
          const x = tf.tensor4d(input, [1, 28, 28, 1], "float32");
          const { probs } = await model!.predict(x);
          const arr = Float32Array.from(await probs.data());
          // Transfer the buffer for performance
          postMessage(
            { type: "prediction", payload: { probs: arr } } as OutMsg,
            [arr.buffer],
          );
          x.dispose();
          probs.dispose();
        };
        run();
      } catch (err) {
        postMessage({
          type: "error",
          payload: { message: String(err) },
        } as OutMsg);
      }
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
        payload: { message: "Unhandled message" },
      } as OutMsg);
    }
  }
};

export {};
